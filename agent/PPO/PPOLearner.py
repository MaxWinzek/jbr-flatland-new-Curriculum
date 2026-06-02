import os

import ray
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from collections import namedtuple
from itertools import count, chain
import time
from copy import deepcopy

from agent.PPO.PPOWorker import PPOWorker
from agent.PPO.PPORunner import PPORunner
from agent.PPO.PPORollout import PPORollout
from agent.PPO.PPOLosses import value_loss, policy_loss, value_loss_with_IS
from agent.judge.Judge import Judge
from logger import log

class PPOLearner():
    def __init__(self, env_config, controller_config, n_workers, device):
        self.n_workers = n_workers
        self.controller = controller_config.create_controller(device)
        self.judge = env_config.env_configs[0].observation_builder_config.timetable_config.create_timetable() # ugly
        self.device = device

        num_gpus = 0
        if device == torch.device("cuda"):
            num_gpus = 1

        ray.init(num_gpus=num_gpus)

        self.curriculum_manager = env_config.create_curriculum_manager()

        self.agents = [None] * n_workers
        for runner_handle in range(n_workers):
            self.agents[runner_handle] = PPOWorker.remote(runner_handle, env_config, controller_config, self.curriculum_manager)
            env_config.update_random_seed()


        self.batch_size = controller_config.batch_size
        self.rollouts_sample = controller_config.rollouts_sample
        self.gae_horizon = controller_config.gae_horizon
        self.value_loss_coeff = controller_config.value_loss_coeff
        self.entropy_coeff = controller_config.entropy_coeff
        self.lam = controller_config.lam
        self.gamma = controller_config.gamma
        self.epochs_update = controller_config.epochs_update
        self.clip_eps = controller_config.clip_eps
        self.optimizer = controller_config.optimizer_config.create_optimizer(
                chain(self.controller.critic_net.parameters(), self.controller.actor_net.parameters()))

        self.train_state = LearnerState(train_iters=2000, exploit_iters=0)

        self.rollouts_buffer = list()

    def save_checkpoint(self, path):
        """Save controller, optimizer, and learner state."""
        os.makedirs(path, exist_ok=True)
        checkpoint_path = os.path.join(path, "checkpoint.pth")
        actor_state, critic_state, target_actor_state = self.controller.get_net_params(device=torch.device('cpu'))
        torch.save({
            'actor_state': actor_state,
            'critic_state': critic_state,
            'target_actor_state': target_actor_state,
            'optimizer_state': self.optimizer.state_dict(),
            'cur_steps': self.train_state.cur_steps,
            'cur_episode': self.train_state.cur_episode,
            'best_exploit_reward': self.train_state.best_exploit_reward
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, path):
        """Load checkpoint if it exists."""
        checkpoint_path = os.path.join(path, "checkpoint.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.controller.update_net_params((
                checkpoint['actor_state'],
                checkpoint['critic_state'],
                checkpoint['target_actor_state']
            ))
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.train_state.cur_steps = checkpoint['cur_steps']
            self.train_state.cur_episode = checkpoint.get('cur_episode', 0)
            self.train_state.best_exploit_reward = checkpoint['best_exploit_reward']
            print(f"Checkpoint loaded from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting fresh.")
            
    # TODO different updates for target_actor/actor
    def _calc_loss(self, state, action, old_log_prob, reward, next_state, done, gae, neighbours_states, actual_len):
        state_values = self.controller.critic_net(state).squeeze(1)
        with torch.no_grad():
            next_state_values = self.controller.critic_net(next_state).squeeze(1)

        logits = self.controller._make_logits(state, neighbours_states)

        action_distribution = Categorical(logits=logits)
        new_log_prob = action_distribution.log_prob(action)

        critic_loss = value_loss_with_IS(state_values, next_state_values, new_log_prob, old_log_prob, reward, done, self.gamma, actual_len)
        actor_loss = policy_loss(gae, new_log_prob, old_log_prob, self.clip_eps)
        entropy_loss = -action_distribution.entropy().mean()

        return critic_loss * self.value_loss_coeff + actor_loss + entropy_loss * self.entropy_coeff


    def _optimize(self, rollout_dict):
        # all agents rollouts combined
        rollouts = [rollout for rollout in rollout_dict.values() if not rollout.is_empty()]
        if not rollouts:
            return
        combined_rollout = PPORollout.combine_rollouts(rollouts)
        self.rollouts_buffer.append(combined_rollout)

        # optimize by batch of several rollouts
        if len(self.rollouts_buffer) != self.rollouts_sample:
            return

        combined_rollout = PPORollout.combine_rollouts(self.rollouts_buffer)
        self.rollouts_buffer.clear()

        state, action, log_prob, reward, next_state, done, neighbours_states, actual_len = combined_rollout.unzip_transitions(self.device)
        gae = combined_rollout.gae.to(self.device)

        for _ in range(self.epochs_update):
            state, action, log_prob, reward, next_state, done, gae, neighbours_states, actual_len = \
                    _permute_all([state, action, log_prob, reward, next_state, done, gae, neighbours_states, actual_len])
            for l in range(0, len(state), self.batch_size):
                r = min(l + self.batch_size, len(state))
                loss = self._calc_loss(state[l:r], action[l:r], log_prob[l:r],
                        reward[l:r], next_state[l:r], done[l:r], gae[l:r], neighbours_states[l:r], actual_len[l:r])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.controller.soft_update(tau=0.05)


    def rollouts(self, max_opt_steps=10**10, max_episodes=10**10):
        log().add_plot("reward", ("train_episode", "train_steps", "reward", "env"))
        log().add_plot("shaped_reward", ("train_episode", "train_steps", "reward", "env"))
        log().add_plot("percent_done", ("train_episode", "train_steps", "percent_done", "env"))
        log().add_plot("time", ("train_episode", "train_steps", "time", "env"))
        log().add_plot("judge_loss", ("episode", "train_steps", "judge_loss", "env"))
        log().add_plot("judge_threshold", ("episode", "train_steps", "judge_threshold", "env"))
        log().add_plot("curriculum_rel_skill", ("train_episode", "train_steps", "rel_skill", "env"))
        log().add_plot("curriculum_mix_progress", ("train_episode", "train_steps", "mix_progress", "env"))
        log().add_plot("curriculum_stage", ("train_episode", "train_steps", "stage", "env"))
        log().add_plot("curriculum_median_history", ("train_episode", "train_steps", "median_history", "env"))
        log().add_plot("curriculum_std_history", ("train_episode", "train_steps", "std_history", "env"))

        controller_params, judge_params = self.controller.get_net_params(device=torch.device("cpu")), \
                self.judge.get_net_params(device=torch.device("cpu"))
        rollouts_list = [agent.run.remote(controller_params, judge_params) for agent in self.agents]
        cur_steps, cur_episode = self.train_state.cur_steps, self.train_state.cur_episode
        while True:
            done_id, rollouts_list = ray.wait(rollouts_list)
            rollout, judge_rollout, info = ray.get(done_id)[0]

            cur_steps += info["steps_done"]
            cur_episode += 1
            self.train_state.cur_episode = cur_episode

            print(cur_episode, info["reward"], info["shaped_reward"], info["env"])
            log().add_plot_point("reward", (cur_episode, cur_steps, info["reward"], info["env"]))
            log().add_plot_point("shaped_reward", (cur_episode, cur_steps, info["shaped_reward"], info["env"]))
            log().add_plot_point("percent_done", (cur_episode, cur_steps, info["percent_done"], info["env"]))
            log().add_plot_point("time", (cur_episode, cur_steps, time.time(), info["env"]))
            sampled_env = info["env"]
            curriculum_stage = int(info.get("_stage", -1)) if not np.isnan(info.get("_stage", np.nan)) else -1
            log().add_plot_point("curriculum_rel_skill", (cur_episode, cur_steps, info.get("rel_skill", np.nan), sampled_env))
            log().add_plot_point("curriculum_mix_progress", (cur_episode, cur_steps, info.get("_mix_progress", np.nan), curriculum_stage))
            log().add_plot_point("curriculum_stage", (cur_episode, cur_steps, info.get("_stage", np.nan), curriculum_stage))
            log().add_plot_point("curriculum_median_history", (cur_episode, cur_steps, info.get("median_history", np.nan), sampled_env))
            log().add_plot_point("curriculum_std_history", (cur_episode, cur_steps, info.get("std_history", np.nan), sampled_env))

            if cur_episode % 100 == 0:
                log().save_logs()
            if cur_episode % 250 == 0:
                self.controller.save_controller(log().get_log_path(), "final_controller.torch")
                self.judge.save_judge(log().get_log_path(), "final_judge.torch")
                self.save_checkpoint(log().get_log_path())
            if self.train_state.is_training():
                self._optimize(rollout)
                #  judge_info = self.judge.optimize(judge_rollout)
                #  log().add_plot_point("judge_loss", (cur_episode, cur_steps, judge_info["loss"], info["env"]))
                #  log().add_plot_point("judge_threshold", (cur_episode, cur_steps, info["judge_threshold"], info["env"]))
            

            self.train_state.step(self.controller, self.judge, info["reward"])
            if cur_steps >= max_opt_steps or cur_episode >= max_episodes:
                break
                
            controller_params, judge_params = self.controller.get_net_params(device=torch.device("cpu")), \
                    self.judge.get_net_params(device=torch.device("cpu"))
            rollouts_list.extend([self.agents[info["handle"]].run.remote(controller_params, judge_params)])

        log().save_logs()
        return

def _permute_all(tensors):
    permutation = torch.randperm(len(tensors[0]))
    for tensor in tensors:
        assert len(tensor) == len(permutation)
    return (tensor[permutation] for tensor in tensors)


class LearnerState():
    def __init__(self, train_iters, exploit_iters):
        self.train_iters = train_iters
        self.exploit_iters = exploit_iters

        self.best_exploit_reward = -np.inf
        self.cur_exploit_reward = 0

        self.cur_steps = 0
        self.cur_episode = 0
        self.train = (train_iters != 0)

    def step(self, controller, judge, reward):
        if self.is_training():
            self._step_train()
        else:
            self._step_exploit(controller, judge, reward)

    def is_training(self):
        return (self.cur_steps % (self.train_iters + self.exploit_iters)) < self.train_iters

    def _step_train(self):
        self.cur_steps += 1

    def _step_exploit(self, controller, judge, reward):
        self.cur_exploit_reward += reward
        self.cur_steps += 1
        if self.is_training(): # end of exploit
            if self.cur_exploit_reward > self.best_exploit_reward:
                self.best_exploit_reward = self.cur_exploit_reward
                controller.save_controller(log().get_log_path())
                judge.save_judge(log().get_log_path())
            self.cur_exploit_reward = 0

