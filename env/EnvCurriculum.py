import numpy as np
import ray

# Skill-based curriculum: gradually shifts probability toward next env once
# rolling avg percent_done exceeds skill_threshold. Avoids hard jumps.
# If regression_threshold is set, drops back a stage when performance collapses
# (only triggers when fully settled on a stage with a full window of samples).
class EnvCurriculumSkill():
    def __init__(self, env_configs, skill_threshold=0.85, window=50, mix_episodes=200, regression_threshold=None):
        self.env_configs = env_configs
        self.skill_threshold = skill_threshold
        self.window = window
        self.mix_episodes = mix_episodes
        self.regression_threshold = regression_threshold

        self._stage = 0          # index of the primary env
        self._mix_progress = 0.0 # 0.0 = all on stage, 1.0 = fully on stage+1
        self._recent_outcomes = []
        self._pos_env = 0

        self.envs = [config.create_env() for config in env_configs]
        self.env = self.envs[0]

    def __getattr__(self, name):
        if name == "cur_env":
            return self._pos_env
        if name == "reset":
            return self._reset
        return getattr(self.env, name)

    def report_outcome(self, percent_done):
        self._recent_outcomes.append(percent_done)
        if len(self._recent_outcomes) > self.window:
            self._recent_outcomes.pop(0)

        avg_skill = sum(self._recent_outcomes) / len(self._recent_outcomes)

        # Regression: only when fully settled on a stage and window is full
        if (self.regression_threshold is not None
                and self._stage > 0
                and self._mix_progress == 0.0
                and len(self._recent_outcomes) >= self.window
                and avg_skill < self.regression_threshold):
            self._stage -= 1
            self._recent_outcomes = []
            return

        if self._stage >= len(self.envs) - 1 and self._mix_progress >= 1.0:
            return
        """
        if self._mix_progress > 0 or avg_skill >= self.skill_threshold:
            if self._stage + 1 < len(self.envs):
                self._mix_progress = min(1.0, self._mix_progress + 1.0 / self.mix_episodes)
                if self._mix_progress >= 1.0:
                    self._stage += 1
                    self._mix_progress = 0.0
                    self._recent_outcomes = []
        """
        if avg_skill >= self.skill_threshold:
            self._mix_progress = min(1.0, self._mix_progress + 1.0 / self.mix_episodes)
            if self._mix_progress >= 1.0:
                self._stage += 1
                self._mix_progress = 0.0
                self._recent_outcomes = []
                
        elif self._mix_progress > 0 and avg_skill < self.regression_threshold:
            self._mix_progress = max(0.0, self._mix_progress - 1.0 / self.mix_episodes)
            
    def _pick_env_index(self):
        if self._stage + 1 < len(self.envs) and self._mix_progress > 0:
            if np.random.random() < self._mix_progress:
                return self._stage + 1
        return self._stage

    def _reset(self):
        self._pos_env = self._pick_env_index()
        self.env = self.envs[self._pos_env]
        return self.env.reset()

import numpy as np


class EnvCurriculumAdaptive:
    def __init__(
        self,
        env_configs,
        window=200,
        mix_episodes=800,           # slower mixing for noisy tasks like Flatland
        alpha=0.02,
        upper_rel=0.92,             # slightly stricter promotion
        lower_rel=0.75,             # slightly wider hysteresis
        min_episodes=150,           # need more evidence
        use_median=True,            # more robust by default
        std_penalty=0.5,            # penalize noisy performance
        promote_patience=20,        # need repeated good signals
        regress_patience=40,        # sustained bad signals before hard regress
        collapse_rel=0.50,          # stronger evidence before hard regress
        fallback_prob=0.35,         # keep more easier-env practice
    ):
        self.env_configs = env_configs
        self.window = window
        self.mix_episodes = mix_episodes
        self.alpha = alpha
        self.upper_rel = upper_rel
        self.lower_rel = lower_rel
        self.min_episodes = min_episodes
        self.use_median = use_median

        self.std_penalty = std_penalty
        self.promote_patience = promote_patience
        self.regress_patience = regress_patience
        self.collapse_rel = collapse_rel
        self.fallback_prob = fallback_prob

        self._stage = 0
        self._mix_progress = 0.0
        self._pos_env = 0

        self._good_streak = 0
        self._bad_streak = 0

        self.envs = [cfg.create_env() for cfg in env_configs]
        self.env = self.envs[0]

        self._env_stats = [
            {
                "ema": 0.0,
                "history": [],
                "count": 0,
            }
            for _ in self.envs
        ]

    def __getattr__(self, name):
        if name == "cur_env":
            return self._pos_env
        if name == "reset":
            return self._reset
        return getattr(self.env, name)

    def _update_stats(self, percent_done):
        stats = self._env_stats[self._pos_env]

        if stats["count"] == 0:
            stats["ema"] = percent_done
        else:
            stats["ema"] = (
                self.alpha * percent_done +
                (1 - self.alpha) * stats["ema"]
            )

        stats["history"].append(percent_done)
        if len(stats["history"]) > self.window:
            stats["history"].pop(0)

        stats["count"] += 1

    def _compute_relative_skill(self):
        stats = self._env_stats[self._pos_env]

        if stats["count"] < self.min_episodes:
            return 0.0

        hist = stats["history"]
        if len(hist) < 10:
            return 0.0

        hist_arr = np.asarray(hist, dtype=np.float32)

        # Robust best estimate
        best = np.percentile(hist_arr, 90)

        # Robust current estimate
        if self.use_median:
            center = float(np.median(hist_arr))
        else:
            center = float(stats["ema"])

        # Flatland-safe tweak:
        # penalize noisy performance so unstable policies do not promote
        noise = float(np.std(hist_arr))
        current = max(0.0, center - self.std_penalty * noise)

        # Optional absolute floor: do not progress if still genuinely weak
        # This prevents advancing just because a poor plateau is "stable"
        if best < 0.3:
            return 0.0

        return current / (best + 1e-6)

    def report_outcome(self, percent_done):
        self._update_stats(percent_done)

        rel_skill = self._compute_relative_skill()

        if self._stage >= len(self.envs) - 1:
            return

        step = 1.0 / self.mix_episodes

        # Use streaks so a few lucky / unlucky episodes do not move curriculum
        if rel_skill >= self.upper_rel:
            self._good_streak += 1
            self._bad_streak = 0
        elif rel_skill < self.lower_rel:
            self._bad_streak += 1
            self._good_streak = 0
        else:
            # In hysteresis gap, decay streaks slightly instead of resetting hard
            self._good_streak = max(0, self._good_streak - 1)
            self._bad_streak = max(0, self._bad_streak - 1)

        # Only change mix after enough consistent evidence
        if self._good_streak >= self.promote_patience:
            self._mix_progress = min(1.0, self._mix_progress + step)
        elif self._bad_streak >= self.promote_patience:
            self._mix_progress = max(0.0, self._mix_progress - step)

        # Advance stage
        if self._mix_progress >= 1.0:
            self._stage += 1
            self._mix_progress = 0.0
            self._good_streak = 0
            self._bad_streak = 0

        # Soft regression first: reduce mix before dropping stage
        if rel_skill < self.collapse_rel:
            self._mix_progress = max(0.0, self._mix_progress - 2 * step)

        # Hard regression only after sustained collapse
        if self._bad_streak >= self.regress_patience and rel_skill < self.collapse_rel and self._stage > 0:
            self._stage -= 1
            self._mix_progress = 0.0
            self._good_streak = 0
            self._bad_streak = 0

    def _pick_env_index(self):
        # Mix current and next env
        if self._stage + 1 < len(self.envs) and self._mix_progress > 0:
            if np.random.random() < self._mix_progress:
                return self._stage + 1

        # Keep more probability of easier env for skill retention
        if self._stage > 0:
            if np.random.random() < self.fallback_prob:
                return self._stage - 1

        return self._stage

    def _reset(self):
        self._pos_env = self._pick_env_index()
        self.env = self.envs[self._pos_env]
        return self.env.reset()


@ray.remote
class AdaptiveCurriculumManager:
    def __init__(
        self,
        num_envs,
        alpha_fast=0.10,
        alpha_slow=0.01,
        stats_window=200,
        std_penalty=0.5,
        min_prob=0.05,
        temperature=0.25,
        progress_weight=0.6,
        underperformance_weight=0.3,
        exploration_weight=0.1,
        unlock_threshold=0.30,
        unlock_min_count=50,
        frontier_lookahead=1,
        fallback_span=1,
        unlock_rel_skill=0.90,
        unlock_rel_skill_by_stage=None,
        unlock_median_floor=0.45,
    ):
        self.num_envs = num_envs
        self.alpha_fast = alpha_fast
        self.alpha_slow = alpha_slow
        self.stats_window = stats_window
        self.std_penalty = std_penalty
        self.min_prob = min_prob
        self.temperature = temperature
        self.progress_weight = progress_weight
        self.underperformance_weight = underperformance_weight
        self.exploration_weight = exploration_weight
        self.unlock_threshold = unlock_threshold
        self.unlock_min_count = unlock_min_count
        self.frontier_lookahead = frontier_lookahead
        self.fallback_span = fallback_span
        self.unlock_rel_skill = unlock_rel_skill
        self.unlock_rel_skill_by_stage = unlock_rel_skill_by_stage
        self.unlock_median_floor = unlock_median_floor

        self._stats = [
            {
                "count": 0,
                "ema_fast": 0.0,
                "ema_slow": 0.0,
                "solve_ema": 0.0,
                "history": [],
            }
            for _ in range(num_envs)
        ]
        self._probs = np.ones(num_envs, dtype=np.float64) / float(num_envs)
        self._anchor = 0

    def _required_rel_skill(self, stage):
        if self.unlock_rel_skill_by_stage:
            idx = min(stage, len(self.unlock_rel_skill_by_stage) - 1)
            return self.unlock_rel_skill_by_stage[idx]
        return self.unlock_rel_skill

    def _update_anchor(self):
        # Keep frontier progression monotonic. Flatland is noisy enough that
        # recomputing unlockability from scratch causes the anchor to collapse
        # backward after temporary dips on already-solved buckets.
        anchor = self._anchor
        while anchor + 1 < self.num_envs:
            stats = self._stats[anchor]
            hist = np.asarray(stats["history"], dtype=np.float32)
            median = float(np.median(hist)) if len(hist) else 0.0
            if len(hist) >= 10:
                best = float(np.percentile(hist, 90))
                std = float(np.std(hist))
                current = max(0.0, median - self.std_penalty * std)
                rel_skill = 0.0 if best < 0.3 else current / (best + 1e-6)
            else:
                rel_skill = 0.0
            if (
                stats["count"] >= self.unlock_min_count and
                stats["ema_slow"] >= self.unlock_threshold and
                median >= self.unlock_median_floor and
                rel_skill >= self._required_rel_skill(anchor)
            ):
                anchor += 1
            else:
                break
        self._anchor = anchor

    def _compute_probs(self):
        scores = np.zeros(self.num_envs, dtype=np.float64)
        self._update_anchor()

        min_allowed = max(0, self._anchor - self.fallback_span)
        max_allowed = min(self.num_envs - 1, self._anchor + self.frontier_lookahead)
        allowed_mask = np.zeros(self.num_envs, dtype=bool)
        allowed_mask[min_allowed:max_allowed + 1] = True

        for idx, stats in enumerate(self._stats):
            if not allowed_mask[idx]:
                scores[idx] = -np.inf
                continue

            count = stats["count"]
            if count == 0:
                progress = 0.0
                solve_ema = 0.0
            else:
                progress = stats["ema_fast"] - stats["ema_slow"]
                solve_ema = stats["solve_ema"]

            # Prefer buckets with ongoing learning progress, but keep pressure
            # on buckets that are not solved yet and a little exploration.
            underperformance = 1.0 - solve_ema
            exploration = 1.0 / np.sqrt(count + 1.0)

            scores[idx] = (
                self.progress_weight * progress +
                self.underperformance_weight * underperformance +
                self.exploration_weight * exploration
            )

        finite_scores = scores[allowed_mask]
        scores[allowed_mask] -= np.max(finite_scores)
        weights = np.exp(scores / max(self.temperature, 1e-6))
        weights[~allowed_mask] = 0.0
        probs = weights / np.sum(weights)

        # Keep a floor on buckets within the active frontier to avoid forgetting.
        probs[allowed_mask] = np.maximum(probs[allowed_mask], self.min_prob)
        probs[~allowed_mask] = 0.0
        probs /= np.sum(probs)
        self._probs = probs

    def sample_env(self):
        return int(np.random.choice(self.num_envs, p=self._probs))

    def report_outcome(self, env_idx, percent_done):
        stats = self._stats[env_idx]

        if stats["count"] == 0:
            stats["ema_fast"] = percent_done
            stats["ema_slow"] = percent_done
            stats["solve_ema"] = 1.0 if percent_done >= 0.999 else 0.0
        else:
            stats["ema_fast"] = (
                self.alpha_fast * percent_done +
                (1.0 - self.alpha_fast) * stats["ema_fast"]
            )
            stats["ema_slow"] = (
                self.alpha_slow * percent_done +
                (1.0 - self.alpha_slow) * stats["ema_slow"]
            )
            stats["solve_ema"] = (
                self.alpha_fast * (1.0 if percent_done >= 0.999 else 0.0) +
                (1.0 - self.alpha_fast) * stats["solve_ema"]
            )

        stats["history"].append(percent_done)
        if len(stats["history"]) > self.stats_window:
            stats["history"].pop(0)
        stats["count"] += 1
        self._compute_probs()

    def get_probs(self):
        return self._probs.tolist()

    def get_env_metrics(self, env_idx):
        stats = self._stats[env_idx]
        hist = np.asarray(stats["history"], dtype=np.float32)

        median = float(np.median(hist)) if len(hist) else np.nan
        std = float(np.std(hist)) if len(hist) else np.nan

        if len(hist) >= 10:
            best = float(np.percentile(hist, 90))
            current = max(0.0, median - self.std_penalty * std)
            rel_skill = 0.0 if best < 0.3 else current / (best + 1e-6)
        else:
            rel_skill = np.nan

        stage = self._anchor
        if stage + 1 < self.num_envs:
            denom = self._probs[stage] + self._probs[stage + 1]
            mix_progress = float(self._probs[stage + 1] / denom) if denom > 0 else 0.0
        else:
            mix_progress = 0.0

        return {
            "rel_skill": rel_skill,
            "_mix_progress": mix_progress,
            "_stage": stage,
            "median_history": median,
            "std_history": std,
        }

    def get_stats(self):
        return [
            {
                "count": stats["count"],
                "ema_fast": stats["ema_fast"],
                "ema_slow": stats["ema_slow"],
                "solve_ema": stats["solve_ema"],
            }
            for stats in self._stats
        ]


class EnvCurriculumAdaptiveDistributed:
    def __init__(self, env_configs, curriculum_manager):
        self.env_configs = env_configs
        self.curriculum_manager = curriculum_manager
        self._pos_env = 0
        self.env = self.env_configs[0].create_env()

    def __getattr__(self, name):
        if name == "cur_env":
            return self._pos_env
        if name == "reset":
            return self._reset
        return getattr(self.env, name)

    def report_outcome(self, percent_done):
        if self.curriculum_manager is not None:
            ray.get(self.curriculum_manager.report_outcome.remote(self._pos_env, percent_done))

    def get_curriculum_metrics(self):
        if self.curriculum_manager is None:
            return {}
        return ray.get(self.curriculum_manager.get_env_metrics.remote(self._pos_env))

    def _reset(self):
        if self.curriculum_manager is not None:
            next_env = ray.get(self.curriculum_manager.sample_env.remote())
        else:
            next_env = np.random.randint(len(self.env_configs))
        if next_env != self._pos_env:
            self._pos_env = next_env
            self.env = self.env_configs[self._pos_env].create_env()
        return self.env.reset()
    
# In Multiagent every worker goes through curriculum
class EnvCurriculum():
    def __init__(self, env_configs, env_episodes):
        self.env_configs = env_configs
        self.env_episodes = env_episodes
        self._pos_env = -1
        self.env = self._start_next()


    def __getattr__(self, name):
        if name == "cur_env":
            return self._pos_env
        if name == "reset":
            return self._reset
        return getattr(self.env, name)


    def _start_next(self):
        if self._pos_env + 1 < len(self.env_configs): # keep launching last env_episode
            self._pos_env += 1
        self._cur_env_episode = 0
        return self.env_configs[self._pos_env].create_env()


    def _reset(self):
        if self._cur_env_episode == self.env_episodes[self._pos_env]:
            self.env = self._start_next()
        self._cur_env_episode += 1
        return self.env.reset()


class EnvCurriculumSample():
    def __init__(self, env_configs, env_probs):
        self.envs = [config.create_env() for config in env_configs]
        self.env_probs = np.array(env_probs, dtype=np.float)
        self.env_probs /= np.sum(self.env_probs)

        self.env = self._start_next()

    def __getattr__(self, name):
        if name == "cur_env":
            return self._pos_env
        if name == "reset":
            return self._reset
        return getattr(self.env, name)

    def _start_next(self):
        self._pos_env = np.random.choice(len(self.envs), p=self.env_probs)
        return self.envs[self._pos_env]

    def _reset(self):
        self.env = self._start_next()
        return self.env.reset()
