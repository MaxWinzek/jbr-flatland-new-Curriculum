import torch
import numpy as np
import random

from env.observations import SimpleObservation

from agent.PPO.PPOLearner import PPOLearner

from configs import Experiment, AdamConfig, FlatlandConfig, \
    SimpleObservationConfig, EnvCurriculumConfig, EnvCurriculumAdaptiveConfig, \
    EnvCurriculumSampleConfig, EnvCurriculumSkillConfig, SimpleRewardConfig, SparseRewardConfig, NearRewardConfig, \
    DeadlockPunishmentConfig,  RewardsComposerConfig, \
    NotStopShaperConfig, FinishRewardConfig, JudgeConfig, LaunchCapConfig
from env.Flatland import Flatland
from agent.judge.Judge import ConstWindowSizeGenerator, LinearOnAgentNumberSizeGenerator
from logger import log, init_logger

from params import FewMoreAgents, PPOParams, SixAgents, TwelveAgents
from params import test_env, PackOfAgents, FewAgents, SeveralAgents, LotsOfAgents, MediumPackOfAgents, HordeOfAgents, \
    SmallPackOfAgents, MediumLotsOfAgents, EighteenAgents, SixteenAgents, ProductionCurriculumEnvs, DenseProductionCurriculumEnvs

def init_random_seeds(RANDOM_SEED, cuda_determenistic):
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    if cuda_determenistic:
        torch.backends.cudnn.deterministic = cuda_determenistic
        torch.backends.cudnn.benchmark = cuda_determenistic

def train_ppo(exp, n_workers):
    init_random_seeds(exp.random_seed, cuda_determenistic=False)
    log().update_params(exp)

    learner = PPOLearner(exp.env_config, exp.controller_config, n_workers, exp.device)

    # Load previous checkpoint if available
    learner.load_checkpoint(log().get_log_path())
    learner.rollouts(max_opt_steps=exp.opt_steps, max_episodes=exp.episodes)
    learner.controller.save_controller(log().get_log_path())



if __name__ == "__main__":
    RANDOM_SEED = 23
    torch.set_printoptions(precision=6, sci_mode=False)
    logname = "run15"
    init_logger("logdir", logname, use_wandb=False)

    timetable_config = LaunchCapConfig(
        initial_fraction=0.4,
        min_active=4,
        cap_increment=1,
        ramp_interval=40,
        min_agents_to_limit=12,
        shuffle=True,
    )

    obs_builder_config = SimpleObservationConfig(max_depth=3, neighbours_depth=3, timetable_config=timetable_config)
    reward_config = RewardsComposerConfig((
        FinishRewardConfig(finish_value=10),
        NearRewardConfig(coeff=0.01),
        DeadlockPunishmentConfig(value=-5),
        NotStopShaperConfig(on_switch_value=0, other_value=0),
    ))
    """
    # Skill-based curriculum: advances when rolling avg percent_done >= threshold,
    # then gradually mixes in the next env over mix_episodes episodes.
    env_config = EnvCurriculumSkillConfig(
        env_configs=[
            FewAgents(RANDOM_SEED),
            FewMoreAgents(RANDOM_SEED),
            SeveralAgents(RANDOM_SEED),
            SixAgents(RANDOM_SEED),
            SmallPackOfAgents(RANDOM_SEED),
            MediumPackOfAgents(RANDOM_SEED),
            PackOfAgents(RANDOM_SEED),
            TwelveAgents(RANDOM_SEED),
            MediumLotsOfAgents(RANDOM_SEED),
            SixteenAgents(RANDOM_SEED),
            EighteenAgents(RANDOM_SEED),
            LotsOfAgents(RANDOM_SEED),
            #HordeOfAgents(RANDOM_SEED),
        ],
        skill_threshold=0.90,
        window=900,
        mix_episodes=800,
        regression_threshold=0.78,
        obs_builder_config=obs_builder_config,
        reward_config=reward_config,
    )
    """

    target_envs = DenseProductionCurriculumEnvs(RANDOM_SEED)
    workers = 4
    exp = Experiment(
        opt_steps=10**10,
        episodes=1605000,
        device=torch.device("cpu"),
        logname=logname,
        random_seed=RANDOM_SEED,
        env_config=EnvCurriculumAdaptiveConfig(
            target_envs,
            alpha_fast=0.10,
            alpha_slow=0.01,
            min_prob=0.08,
            temperature=0.20,
            progress_weight=0.55,
            underperformance_weight=0.35,
            exploration_weight=0.10,
            std_penalty=0.35,
            unlock_threshold=0.45,
            unlock_min_count=150,
            frontier_lookahead=1,
            fallback_span=1,
            unlock_rel_skill=0.88,
            unlock_rel_skill_by_stage=[0.90, 0.90, 0.89, 0.89, 0.88, 0.88, 0.87, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81, 0.80],
            unlock_median_floor=0.50,
            obs_builder_config=obs_builder_config,
            reward_config=reward_config,
        ),
        controller_config=PPOParams(
            rollouts_sample=8,
            clip_eps=0.15,
        ),

    )
    train_ppo(exp, workers)


