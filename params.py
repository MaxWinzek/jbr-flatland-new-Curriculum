from collections import namedtuple
import math
import copy

from configs import AdamConfig, PPOConfig, \
        FlatlandConfig, SimpleObservationConfig, SimpleRewardConfig, SparseRewardConfig


PRODUCTION_MALFUNCTION_RATE = 1. / 300


def PPOParams(
        lr=2e-4,
        batch_size=128,
        rollouts_sample=32,
        gae_horizon=256,
        epochs_update=3,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.1,
        value_loss_coeff=0.5,
        entropy_coeff=0.02,
        actor_layers_sz=None,
        critic_layers_sz=None,
    ):
    if actor_layers_sz is None:
        # Keep the legacy checkpoint-compatible action head by default.
        actor_layers_sz = [256]
    if critic_layers_sz is None:
        # An empty head preserves the legacy critic architecture.
        critic_layers_sz = []
    return PPOConfig(
        state_sz = 203, # TODO pass approprietly
        action_sz = 3,
        neighbours_depth = 3,
        optimizer_config = AdamConfig(lr=lr),
        batch_size = batch_size,
        rollouts_sample = rollouts_sample,
        gae_horizon = gae_horizon,
        epochs_update = epochs_update,
        gamma = gamma,
        lam = lam,
        clip_eps = clip_eps,
        value_loss_coeff = value_loss_coeff,
        entropy_coeff = entropy_coeff,
        actor_layers_sz=actor_layers_sz,
        critic_layers_sz=critic_layers_sz,
    )
#256, 128
# env configs


def _flatland_env(
        random_seed,
        height,
        width,
        n_agents,
        n_cities,
        malfunction_rate=PRODUCTION_MALFUNCTION_RATE,
    ):
    return FlatlandConfig(
            height=height,
            width=width,
            n_agents=n_agents,
            n_cities=n_cities,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=malfunction_rate,
            random_seed=random_seed,
            greedy=True,
        )

def FewAgents(random_seed):
    return FlatlandConfig(
            height=35,
            width=35,
            n_agents=2,
            n_cities=2,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./100,
            random_seed=random_seed,
            greedy=True,
        )

def FewMoreAgents(random_seed):
    return FlatlandConfig(
            height=35,
            width=35,
            n_agents=4,
            n_cities=2,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./100,
            random_seed=random_seed,
            greedy=True,
        )
def SeveralAgents(random_seed):
    return FlatlandConfig(
            height=35,
            width=35,
            n_agents=5,
            n_cities=3,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./200,
            random_seed=random_seed,
            greedy=True,
        )

def SixAgents(random_seed):  # new env between 2 and 3
    return _flatland_env(random_seed, height=35, width=35, n_agents=6, n_cities=3)


def SmallPackOfAgents(random_seed):
    return _flatland_env(random_seed, height=35, width=35, n_agents=7, n_cities=3)

def MediumPackOfAgents(random_seed):
    return _flatland_env(random_seed, height=40, width=40, n_agents=8, n_cities=3)

def NineAgents(random_seed):
    return _flatland_env(random_seed, height=42, width=40, n_agents=9, n_cities=4)

def PackOfAgents(random_seed):
    return _flatland_env(random_seed, height=45, width=40, n_agents=10, n_cities=4)

def ElevenAgents(random_seed):
    return _flatland_env(random_seed, height=45, width=40, n_agents=11, n_cities=4)

def TwelveAgents(random_seed):
    return _flatland_env(random_seed, height=45, width=40, n_agents=12, n_cities=4)

def ThirteenAgents(random_seed):
    return _flatland_env(random_seed, height=48, width=50, n_agents=13, n_cities=5)

def MediumLotsOfAgents(random_seed):
    return _flatland_env(random_seed, height=50, width=60, n_agents=14, n_cities=5)

def SixteenAgents(random_seed):
    return _flatland_env(random_seed, height=52, width=65, n_agents=16, n_cities=5)

def SeventeenAgents(random_seed):
    return _flatland_env(random_seed, height=53, width=68, n_agents=17, n_cities=5)

def EighteenAgents(random_seed):
    return _flatland_env(random_seed, height=54, width=70, n_agents=18, n_cities=5)

def LotsOfAgents(random_seed):
    return _flatland_env(random_seed, height=55, width=75, n_agents=20, n_cities=6)

def HordeOfAgents(random_seed):
    return FlatlandConfig(
            height=120,
            width=80,
            n_agents=50,
            n_cities=10,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./1000,
            random_seed=random_seed,
            greedy=True,
        )


def FifteenAgents(random_seed):
    return _flatland_env(random_seed, height=51, width=62, n_agents=15, n_cities=5)


def NineteenAgents(random_seed):
    return _flatland_env(random_seed, height=55, width=72, n_agents=19, n_cities=6)


def ProductionCurriculumEnvs(random_seed, include_bridges=False):
    envs = [
        SixAgents(random_seed),
        SmallPackOfAgents(random_seed),
        MediumPackOfAgents(random_seed),
        PackOfAgents(random_seed),
        TwelveAgents(random_seed),
        MediumLotsOfAgents(random_seed),
    ]
    if include_bridges:
        envs.append(FifteenAgents(random_seed))
    envs.extend([
        SixteenAgents(random_seed),
        EighteenAgents(random_seed),
    ])
    if include_bridges:
        envs.append(NineteenAgents(random_seed))
    envs.append(LotsOfAgents(random_seed))
    return envs


def DenseProductionCurriculumEnvs(random_seed):
    return [
        SixAgents(random_seed),
        SmallPackOfAgents(random_seed),
        MediumPackOfAgents(random_seed),
        NineAgents(random_seed),
        PackOfAgents(random_seed),
        ElevenAgents(random_seed),
        TwelveAgents(random_seed),
        ThirteenAgents(random_seed),
        MediumLotsOfAgents(random_seed),
        FifteenAgents(random_seed),
        SixteenAgents(random_seed),
        SeventeenAgents(random_seed),
        EighteenAgents(random_seed),
        NineteenAgents(random_seed),
        LotsOfAgents(random_seed),
    ]


def env6(random_seed):
    return FlatlandConfig(
            height=40,
            width=60,
            n_agents=80,
            n_cities=9,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./800,
            random_seed=random_seed,
            greedy=True,
        )


def env7(random_seed):
    return FlatlandConfig(
            height=60,
            width=40,
            n_agents=80,
            n_cities=13,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./800,
            random_seed=random_seed,
            greedy=True,
        )


def env8(random_seed):
    return FlatlandConfig(
            height=60,
            width=60,
            n_agents=80,
            n_cities=17,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./800,
            random_seed=random_seed,
            greedy=True,
        )

def env9(random_seed):
    return FlatlandConfig(
            height=80,
            width=120,
            n_agents=100,
            n_cities=21,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./1000,
            random_seed=random_seed,
            greedy=True,
        )

def env10(random_seed):
    return FlatlandConfig(
            height=100,
            width=80,
            n_agents=100,
            n_cities=25,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./1000,
            random_seed=random_seed,
            greedy=True,
        )

def env11(random_seed):
    return FlatlandConfig(
            height=100,
            width=100,
            n_agents=200,
            n_cities=29,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./2000,
            random_seed=random_seed,
            greedy=True,
        )

def env12(random_seed):
    return FlatlandConfig(
            height=150,
            width=150,
            n_agents=200,
            n_cities=33,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./2000,
            random_seed=random_seed,
            greedy=True,
        )

def env13(random_seed):
    return FlatlandConfig(
            height=150,
            width=150,
            n_agents=400,
            n_cities=37,
            grid_distribution_of_cities=False,
            max_rails_between_cities=2,
            max_rail_in_cities=4,
            observation_builder_config=None,
            reward_config=None,
            malfunction_rate=1./4000,
            random_seed=random_seed,
            greedy=True,
        )

_test_env = list()
for i in range(50):
    if not i:
        n_agents = 1
        n_cities = 2
        x_dim = 25
        y_dim = 25
    else:
        n_agents = n_agents + math.ceil(10 ** (len(str(n_agents)) - 1)* 0.75)
        n_cities = n_agents//10 + 2
        x_dim = math.ceil(math.sqrt(150 * n_cities)) + 7
        y_dim = x_dim

    _test_env.append(FlatlandConfig(
        height=x_dim,
        width=y_dim,
        n_agents=n_agents,
        n_cities=n_cities,
        grid_distribution_of_cities=False,
        max_rails_between_cities=2,
        max_rail_in_cities=4,
        observation_builder_config=None,
        reward_config=None,
        malfunction_rate=1./1500,
        random_seed=None,
        greedy=True,
    ))


def test_env(random_seed, i):
    env = _test_env[i]
    env.random_seed = random_seed
    return env
