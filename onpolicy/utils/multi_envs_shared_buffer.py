import numpy as np
from copy import deepcopy
import torch

from onpolicy.utils.shared_buffer import SharedReplayBuffer, RNN_Cognition_Buffer, subtask_Buffer
from onpolicy.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv, ShareVecEnv, CloudpickleWrapper
from multiprocessing import Process, Pipe
from onpolicy.envs.starcraft2.feature_translation import find_map_ohid

def envshareworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    vecenv = env_fn_wrapper.x
    while True:
        cmd, data, env_idx = remote.recv()
        if cmd == 'step':
            # print(f"step: {env_idx} | action: {data}")
            ob, s_ob, reward, done, info, available_actions = vecenv.step(data)
            # 貌似不需要下边这个if...else...用来reset
            # 因为这里vecenv实际上是类ShareSubprocVecEnv, 其本身有实现reset
            # 但是加了好像也影响不大
            # if 'bool' in done.__class__.__name__:
            #     if done:
            #         ob, s_ob, available_actions = vecenv.reset()
            # else:
            #     if np.all(done):
            #         ob, s_ob, available_actions = vecenv.reset()

            remote.send((ob, s_ob, reward, done, info, available_actions, env_idx))
        elif cmd == 'reset':
            ob, s_ob, available_actions = vecenv.reset()
            remote.send((ob, s_ob, available_actions, env_idx))
        elif cmd == 'reset_task':
            ob = vecenv.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            vecenv.close()
            remote.close()
            break
        elif cmd == 'save_replay':
            print("TEST: save replay")
            vecenv.save_replay()
        else:
            raise NotImplementedError

class MultiEnvShareSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns_list, num_threads_per_env, multi_envs, n_agents_list, n_enemies_list, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        if num_threads_per_env == 1:
            self.env_list = [
                ShareDummyVecEnv([env_fns]) for env_fns in env_fns_list
            ]
        else:
            self.env_list = [
                ShareSubprocVecEnv(env_fns) for env_fns in env_fns_list
            ]

        self.waiting = False
        self.closed = False
        n_multi_envs = len(env_fns_list)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_multi_envs)])
        self.ps = [Process(target=envshareworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, self.env_list)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.num_thread_per_env = num_threads_per_env
        self.multi_envs = multi_envs
        self.n_agents_list = n_agents_list
        self.n_enemies_list = n_enemies_list
        self.observation_space, self.share_observation_space, self.action_space = [], [], []
        current_threads = 0
        for idx, _ in enumerate(multi_envs):
            ob_sp, sh_ob_sp, ac_sp = self.env_list[idx].observation_space, \
                self.env_list[idx].share_observation_space, \
                    self.env_list[idx].action_space
            self.observation_space.append(ob_sp)
            self.share_observation_space.append(sh_ob_sp)
            self.action_space.append(ac_sp)
            current_threads += num_threads_per_env
            print(f"MultiTrainEnvShareSubprocVecEnv init {idx}")

    # def step(self, actions, idx):
    #     return self.env_list[idx].step(actions)

    def save_replay(self):
        for idx, remote in enumerate(self.remotes):
            remote.send(('save_replay', None, idx))

    def step_async(self, actions_list):
        for idx, (remote, action) in enumerate(zip(self.remotes, actions_list)):
            remote.send(('step', action, idx))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions, idxs = zip(*results)
        # obs is a tuple consisting of obs corronspanding to different env in multi envs
        return obs, share_obs, rews, dones, infos, available_actions, idxs

    def reset(self):
        for idx, remote in enumerate(self.remotes):
            remote.send(('reset', None, idx))
        self.waiting = True
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, available_actions, idxs = zip(*results)
        # return np.stack(obs), np.stack(share_obs), np.stack(available_actions)
        return obs, share_obs, available_actions, idxs

    # def reset_task(self):
    #     for remote in self.remotes:
    #         remote.send(('reset_task', None))
    #     return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None, None))
        for p in self.ps:
            p.join()
        self.closed = True


class MultiEnvSharedReplayBuffer(SharedReplayBuffer):
    def __init__(self, args, 
                 num_agents_list, 
                 num_enemies_list, 
                 obs_space_list, 
                 cent_obs_space_list, 
                 act_space_list, 
                 num_thread_per_env):
        self.buffer_lists = []
        for n_agents, n_enemies, ob_space, cent_ob_space, ac_space in zip(num_agents_list, num_enemies_list, obs_space_list, cent_obs_space_list, act_space_list):
            args_copy = deepcopy(args)
            args_copy.n_rollout_threads = num_thread_per_env
            if args_copy.algorithm_name == "rnn_cognition" or args_copy.algorithm_name == "rnn_cognition_transfer":
                self.buffer_lists.append(
                    RNN_Cognition_Buffer(args_copy, n_agents, n_enemies, ob_space[0], cent_ob_space[0], ac_space[0])
                )
            elif args_copy.algorithm_name == "subtask" or args_copy.algorithm_name == "subtask_transfer":
                self.buffer_lists.append(
                    subtask_Buffer(args_copy, n_agents, ob_space[0], cent_ob_space[0], ac_space[0])
                )
            else:
                self.buffer_lists.append(
                    SharedReplayBuffer(
                        args_copy, n_agents, ob_space[0], cent_ob_space[0], ac_space[0]
                    ) 
                ) 
        self.num_thread_per_env = num_thread_per_env
        self.n_agents_list = num_agents_list
        self.n_enemies_list = num_enemies_list
        self.num_multi_envs = len(self.n_agents_list)

    def split_data(self, *data):
        data_lists = []
        for data_item in data:
            data_lists.append(np.split(data_item, self.num_multi_envs, axis=0))
        return data_lists

    def insert(self, idx, share_obs, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs, 
               value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        self.buffer_lists[idx].insert(
            share_obs, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs, 
            value_preds, rewards, masks, bad_masks, active_masks, available_actions
        )
    
    def after_update(self):
        for idx in range(len(self.n_agents_list)):
            self.buffer_lists[idx].after_update()

    def compute_returns(self, idx, next_value, value_normalizer=None):
        self.buffer_lists[idx].compute_returns(next_value, value_normalizer)
    
    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        for idx, env_advantages in enumerate(advantages):
            yield self.buffer_lists[idx].feed_forward_generator(env_advantages, num_mini_batch, mini_batch_size)

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        for idx, env_advantages in enumerate(advantages):
            yield self.buffer_lists[idx].naive_recurrent_generator(env_advantages, num_mini_batch)

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        for idx, env_advantages in enumerate(advantages):
            yield self.buffer_lists[idx].recurrent_generator(env_advantages, num_mini_batch, data_chunk_length)
