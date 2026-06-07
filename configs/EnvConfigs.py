from env.Flatland import Flatland, FlatlandWrapper
from env.GreedyFlatland import GreedyFlatland
from env.CartPole import MultiCartPole
from env.LunarLander import MultiLunarLander
from env.EnvCurriculum import EnvCurriculum, EnvCurriculumSample, EnvCurriculumSkill, EnvCurriculumAdaptiveDistributed, AdaptiveCurriculumManager

from configs.Config import Config

class EnvConfig(Config):
    def __init__(self):
        pass

    def create_env(self, curriculum_manager=None):
        pass

    def create_curriculum_manager(self):
        return None


class FlatlandConfig(EnvConfig):
    def __init__(
            self,
            height,
            width,
            n_agents,
            n_cities,
            grid_distribution_of_cities,
            max_rails_between_cities,
            max_rail_in_cities,
            observation_builder_config,
            reward_config,
            malfunction_rate,
            greedy,
            random_seed,
        ):
        super(FlatlandConfig, self).__init__()
        self.height = height
        self.width = width
        self.n_agents = n_agents
        self.n_cities = n_cities
        self.grid_distribution_of_cities = grid_distribution_of_cities
        self.max_rails_between_cities = max_rails_between_cities
        self.max_rail_in_cities = max_rail_in_cities
        self.observation_builder_config = observation_builder_config
        self.reward_config = reward_config
        self.malfunction_rate = malfunction_rate
        self.random_seed = random_seed
        self.greedy = greedy

    def update_random_seed(self):
        self.random_seed += 1

    def set_obs_builder_config(self, obs_builder_config):
        self.observation_builder_config = obs_builder_config

    def set_reward_config(self, reward_config):
        self.reward_config = reward_config

    def create_env(self, curriculum_manager=None):
        obs_builder = self.observation_builder_config.create_builder()
        reward_shaper = self.reward_config.create_reward_shaper()
        rail_env = FlatlandWrapper(Flatland(
            height=self.height,
            width=self.width,
            n_agents=self.n_agents,
            n_cities=self.n_cities,
            grid_distribution_of_cities=self.grid_distribution_of_cities,
            max_rails_between_cities=self.max_rails_between_cities,
            max_rail_in_cities=self.max_rail_in_cities,
            observation_builder=obs_builder,
            malfunction_rate=self.malfunction_rate,
            random_seed=self.random_seed,
        ), reward_shaper=reward_shaper)
        if self.greedy:
            rail_env = GreedyFlatland(rail_env)
        return rail_env


class EnvCurriculumConfig(EnvConfig):
    def __init__(self, env_configs, env_episodes, obs_builder_config=None, reward_config=None):
        self.env_configs = env_configs
        self.env_episodes = env_episodes

        if obs_builder_config is not None:
            self.set_obs_builder_config(obs_builder_config)

        if reward_config is not None:
            self.set_reward_config(reward_config)

    def update_random_seed(self):
        for conf in self.env_configs:
            conf.update_random_seed()

    def set_obs_builder_config(self, obs_builder_config):
        for conf in self.env_configs:
            conf.set_obs_builder_config(obs_builder_config)

    def set_reward_config(self, reward_config):
        for conf in self.env_configs:
            conf.set_reward_config(reward_config)

    def create_env(self, curriculum_manager=None):
        return EnvCurriculum(self.env_configs, self.env_episodes)


class EnvCurriculumSampleConfig(EnvConfig):
    def __init__(self, env_configs, env_probs, obs_builder_config=None, reward_config=None):
        self.env_configs = env_configs
        self.env_probs = env_probs

        if obs_builder_config is not None:
            self.set_obs_builder_config(obs_builder_config)

        if reward_config is not None:
            self.set_reward_config(reward_config)

    def update_random_seed(self):
        for conf in self.env_configs:
            conf.update_random_seed()

    def set_obs_builder_config(self, obs_builder_config):
        for conf in self.env_configs:
            conf.set_obs_builder_config(obs_builder_config)

    def set_reward_config(self, reward_config):
        for conf in self.env_configs:
            conf.set_reward_config(reward_config)

    def create_env(self, curriculum_manager=None):
        return EnvCurriculumSample(self.env_configs, self.env_probs)


class EnvCurriculumSkillConfig(EnvConfig):
    def __init__(self, env_configs, skill_threshold=0.85, window=50, mix_episodes=200,
                 regression_threshold=None, obs_builder_config=None, reward_config=None):
        self.env_configs = env_configs
        self.skill_threshold = skill_threshold
        self.window = window
        self.mix_episodes = mix_episodes
        self.regression_threshold = regression_threshold

        if obs_builder_config is not None:
            self.set_obs_builder_config(obs_builder_config)
        if reward_config is not None:
            self.set_reward_config(reward_config)

    def update_random_seed(self):
        for conf in self.env_configs:
            conf.update_random_seed()

    def set_obs_builder_config(self, obs_builder_config):
        for conf in self.env_configs:
            conf.set_obs_builder_config(obs_builder_config)

    def set_reward_config(self, reward_config):
        for conf in self.env_configs:
            conf.set_reward_config(reward_config)

    def create_env(self, curriculum_manager=None):
        return EnvCurriculumSkill(self.env_configs, self.skill_threshold, self.window, self.mix_episodes, self.regression_threshold)


class EnvCurriculumAdaptiveConfig(EnvConfig):
    def __init__(
        self,
        env_configs,
        random_seed=None,
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
        obs_builder_config=None,
        reward_config=None,
    ):
        self.env_configs = env_configs
        self.random_seed = random_seed
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

        if obs_builder_config is not None:
            self.set_obs_builder_config(obs_builder_config)
        if reward_config is not None:
            self.set_reward_config(reward_config)

    def update_random_seed(self):
        for conf in self.env_configs:
            conf.update_random_seed()

    def set_obs_builder_config(self, obs_builder_config):
        for conf in self.env_configs:
            conf.set_obs_builder_config(obs_builder_config)

    def set_reward_config(self, reward_config):
        for conf in self.env_configs:
            conf.set_reward_config(reward_config)

    def create_curriculum_manager(self):
        return AdaptiveCurriculumManager.remote(
            len(self.env_configs),
            random_seed=self.random_seed,
            alpha_fast=self.alpha_fast,
            alpha_slow=self.alpha_slow,
            stats_window=self.stats_window,
            std_penalty=self.std_penalty,
            min_prob=self.min_prob,
            temperature=self.temperature,
            progress_weight=self.progress_weight,
            underperformance_weight=self.underperformance_weight,
            exploration_weight=self.exploration_weight,
            unlock_threshold=self.unlock_threshold,
            unlock_min_count=self.unlock_min_count,
            frontier_lookahead=self.frontier_lookahead,
            fallback_span=self.fallback_span,
            unlock_rel_skill=self.unlock_rel_skill,
            unlock_rel_skill_by_stage=self.unlock_rel_skill_by_stage,
            unlock_median_floor=self.unlock_median_floor,
        )

    def create_env(self, curriculum_manager=None):
        return EnvCurriculumAdaptiveDistributed(self.env_configs, curriculum_manager)


class CartPoleConfig(EnvConfig):
    def __init__(self, n_agents, random_seed):
        self.n_agents = n_agents
        self.random_seed = random_seed

    def update_random_seed(self):
        self.random_seed += 1

    def create_env(self, curriculum_manager=None):
        return MultiCartPole(self.n_agents, self.random_seed)


class LunarLanderConfig(EnvConfig):
    def __init__(self, n_agents, random_seed):
        self.n_agents = n_agents
        self.random_seed = random_seed

    def update_random_seed(self):
        self.random_seed += 1

    def create_env(self, curriculum_manager=None):
        return MultiLunarLander(self.n_agents, self.random_seed)
