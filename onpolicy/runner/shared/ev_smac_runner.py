import time
import wandb
import numpy as np
from functools import reduce
import os
from tensorboardX import SummaryWriter
import torch
from onpolicy.runner.shared.base_runner import Runner
from onpolicy.utils.multi_envs_shared_buffer import MultiEnvSharedReplayBuffer
from onpolicy.algorithms.r_mappo.ev_r_mappo import EntityVAER_MAPPO as TrainAlgo
from onpolicy.algorithms.r_mappo.algorithm.ev_rMAPPOPolicy import EntityVAER_MAPPOPolicy as Policy
from onpolicy.runner.shared.smac_runner import _t2n
import json

class EntityVAESMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        print("--------------------------------------------------------------------------------------------")
        print("=======================------------Entity VAE SMAC Runner-----------========================")
        print("--------------------------------------------------------------------------------------------")

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.latent_dim = self.all_args.latent_dim

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            self.trajectory_dir = str(self.run_dir / 'trajectory')
            if not os.path.exists(self.trajectory_dir):
                os.makedirs(self.trajectory_dir)
            
            # save hyperparameters
            with open(os.path.join(str(self.run_dir), "args.json"), "wt") as f:
                json.dump(vars(self.all_args), f, indent=4) # Indent 4 spaces in JSON format

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        self.multi_envs = self.envs.multi_envs
        self.num_multi_envs = len(self.multi_envs)
        self.num_thread_per_env = self.envs.num_thread_per_env
        self.num_agents_list = self.envs.n_agents_list
        self.num_enemies_list = self.envs.n_enemies_list
        self.obs_space_list = self.envs.observation_space
        self.cent_obs_space_list = self.envs.share_observation_space
        self.act_space_list = self.envs.action_space

        # policy network
        self.policy = Policy(self.all_args,
                            self.multi_envs,
                            self.num_thread_per_env,
                            self.num_agents_list,
                            self.num_enemies_list,
                            self.envs.observation_space[0],
                            share_observation_space,
                            self.envs.action_space[0],
                            device = self.device)

        
        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, self.num_agents_list, self.num_enemies_list, device = self.device)
        if self.model_dir is not None:
            if self.all_args.transfer_only_context_encoder:
                self.restore_only_con_encoder()
            else:
                self.restore()
        
        # self.extra_rl_posterior = self.all_args.extra_rl_posterior
        # self.num_steps_prior = self.episode_length
        # self.num_steps_posterior = self.episode_length
        # self.num_extra_rl_steps_posterior = self.episode_length if self.extra_rl_posterior else 0 # self.episode_length if off-policy
        # assert self.num_extra_rl_steps_posterior == 0, f"Now just compatible with num_extra_rl_steps_posterior = 0"
        # self.update_post_train = 1
        
        # Buffer
        # self.buffer_length_repeats = 2 if self.num_extra_rl_steps_posterior == 0 else 3
        # Buffer
        self.buffer = MultiEnvSharedReplayBuffer(
            self.all_args, self.num_agents_list, self.num_enemies_list,
            self.obs_space_list, self.cent_obs_space_list, self.act_space_list, 
            self.num_thread_per_env
        )
        

    def run(self):
        # we only evaluation the model from subtask
        if self.model_dir is not None:
            if self.all_args.only_evaluate or self.all_args.save_replay:
                self.evaluate4replay()
                return
        # training
        if "subtask" in self.algorithm_name:
            self.warmup4subtask()
        else:
            self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = [np.zeros(self.num_thread_per_env, dtype=np.float32) for _ in range(self.num_multi_envs)]
        last_battles_won = [np.zeros(self.num_thread_per_env, dtype=np.float32) for _ in range(self.num_multi_envs)]

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            if "subtask" in self.algorithm_name:
                infos_list = self.collect_data4subtask(self.episode_length)
            else:
                infos_list = self.collect_data(self.episode_length)
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads          
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.multi_envs,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "StarCraft2":
                    for idx, infos in enumerate(infos_list):
                        battles_won = []
                        battles_game = []
                        incre_battles_won = []
                        incre_battles_game = []                    

                        for i, info in enumerate(infos):
                            if 'battles_won' in info[0].keys():
                                battles_won.append(info[0]['battles_won'])
                                incre_battles_won.append(info[0]['battles_won']-last_battles_won[idx][i])
                            if 'battles_game' in info[0].keys():
                                battles_game.append(info[0]['battles_game'])
                                incre_battles_game.append(info[0]['battles_game']-last_battles_game[idx][i])

                        incre_win_rate = np.sum(incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game)>0 else 0.0
                        print("{} incre win rate is {}.".format(self.multi_envs[idx], incre_win_rate))
                        if self.use_wandb:
                            wandb.log({"incre_win_rate_{}".format(self.multi_envs[idx]): incre_win_rate}, step=total_num_steps)
                        else:
                            self.writter.add_scalars("incre_win_rate", {"incre_win_rate_{}".format(self.multi_envs[idx]): incre_win_rate}, total_num_steps)
                        
                        last_battles_game[idx] = battles_game
                        last_battles_won[idx] = battles_won

                dead_ratio = []
                for idx in range(self.num_multi_envs):
                    dead_ratio.append(1 - self.buffer.buffer_lists[idx].active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer.buffer_lists[idx].active_masks.shape)))
                train_infos['dead_ratio'] = dead_ratio
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                # self.eval(total_num_steps)
                # entity_product已废弃
                if self.all_args.record_attention:
                    self.eval4attentraj(total_num_steps)
                elif self.algorithm_name == "entity_product" or self.algorithm_name == "entity_product_transfer" or \
                    self.algorithm_name == "subtask" or self.algorithm_name == "subtask_transfer":
                    self.eval4traj(total_num_steps)
                else:
                    self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs_tuple, share_obs_tuple, available_actions_tuple, idxs_tuple = self.envs.reset()

        for obs, share_obs, available_actions, idx in zip(obs_tuple, share_obs_tuple, available_actions_tuple, idxs_tuple):

            # replay buffer
            if not self.use_centralized_V:
                share_obs = obs

            self.buffer.buffer_lists[idx].share_obs[0] = share_obs.copy()
            self.buffer.buffer_lists[idx].obs[0] = obs.copy()
            self.buffer.buffer_lists[idx].available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        # (threads*na, _size)
        cent_obs = [np.concatenate(self.buffer.buffer_lists[idx].share_obs[step]) for idx in range(self.num_multi_envs)]
        obs = [np.concatenate(self.buffer.buffer_lists[idx].obs[step]) for idx in range(self.num_multi_envs)]
        rnn_state = [np.concatenate(self.buffer.buffer_lists[idx].rnn_states[step]) for idx in range(self.num_multi_envs)]
        rnn_state_critic = [np.concatenate(self.buffer.buffer_lists[idx].rnn_states_critic[step]) for idx in range(self.num_multi_envs)]
        masks = [np.concatenate(self.buffer.buffer_lists[idx].masks[step]) for idx in range(self.num_multi_envs)]
        available_actions = [np.concatenate(self.buffer.buffer_lists[idx].available_actions[step]) for idx in range(self.num_multi_envs)]
        th_na_list = [cent_ob.shape[0] for cent_ob in cent_obs]
        
        value, action, action_log_prob, rnn_state, rnn_state_critic = \
            self.trainer.policy.get_actions(
                cent_obs, obs, rnn_state, rnn_state_critic, masks, available_actions, 
                n_agents=self.num_agents_list, 
                n_enemies=self.num_enemies_list,
            )

        values = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(value, th_na_list, dim=0)]
        actions = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(action, th_na_list, dim=0)]
        action_log_probs = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(action_log_prob, th_na_list, dim=0)]
        # 应该是用不到这个函数
        if self.all_args.use_naive_recurrent_policy or self.all_args.use_recurrent_policy:
            # 多个环境的rnn_state被放到了一个列表中
            rnn_states = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(rnn_state, th_na_list, dim=0)]
            rnn_states_critic = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(rnn_state_critic, th_na_list, dim=0)]
        else:
            # rnn_state仍是tensor而不是若干个tensor组成的list
            rnn_states = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in rnn_state]
            rnn_states_critic = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in rnn_state_critic]
        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data, idx):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        n_agents = self.num_agents_list[idx]
        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), n_agents, *self.buffer.buffer_lists[idx].rnn_states.shape[3:]), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), n_agents, *self.buffer.buffer_lists[idx].rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.num_thread_per_env, n_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), n_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.num_thread_per_env, n_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), n_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(n_agents)] for info in infos])
        
        if not self.use_centralized_V:
            share_obs = obs
        
        self.buffer.insert(idx, share_obs, obs, rnn_states, rnn_states_critic,
                            actions, action_log_probs, values, rewards, masks, bad_masks, active_masks, available_actions)

    def collect_data(self, num_samples):
        for step in range(num_samples):
            # Sample actions
            values_list, actions_list, action_log_probs_list, \
                rnn_states_list, rnn_states_critic_list = self.collect(step)
            # Obser reward and next obs
            obs_tuple, share_obs_tuple, rewards_tuple, dones_tuple, \
                infos_tuple, available_actions_tuple, idxs_tuple = self.envs.step(actions_list)
            
            infos_list = [None for _ in idxs_tuple]
            
            for obs, share_obs, rewards, dones, infos, available_actions, \
                values, actions, action_log_probs, rnn_states, rnn_states_critic, idx in zip(
                    obs_tuple, share_obs_tuple, rewards_tuple, dones_tuple, infos_tuple, available_actions_tuple, \
                        values_list, actions_list, action_log_probs_list, rnn_states_list, rnn_states_critic_list, idxs_tuple
                ):
                data = obs, share_obs, rewards, dones, infos, available_actions, \
                    values, actions, action_log_probs, \
                    rnn_states, rnn_states_critic 
                
                # insert data into buffer/enc_buffer
                self.insert(data, idx)
                # infos
                infos_list[idx] = infos
                
                # done_idxs = np.where(np.all(dones, axis=1))[0].tolist()
                # # 如果有环境done了, 则判断对done的环境是否重新sample z
                # if done_idxs != [] and resample_z_rate != np.inf:
                #     # print(f"Env: {idx} | SubEnv: {done_idxs} is done; Then rsmaple z")
                #     self.policy.sample_z(multi_envs_id=idx, env_ids=done_idxs)
                # # 如果有环境done了, 则判断对done的环境是否更新z的后验
                # if done_idxs != [] and update_posterior_rate != np.inf:
                #     context = self.policy.sample_context(multi_envs_id=idx, enc_buffer=self.enc_buffer, env_ids=done_idxs, end_idx=update_posterior_end_idx)
                #     self.policy.infer_posterior(multi_envs_id=idx, env_ids=done_idxs, context=context)

        return infos_list

    def log_train(self, train_infos, total_num_steps):
        average_step_rewrads = [np.mean(self.buffer.buffer_lists[idx].rewards) for idx in range(self.num_multi_envs)]
        train_infos["average_step_rewards"] = average_step_rewrads
        for k, v in train_infos.items():
            if self.use_wandb:
                if isinstance(v, list):
                    for idx, env_name in enumerate(self.multi_envs):
                        wandb.log(k, {"{}_{}".format(k, env_name): v[idx]}, total_num_steps)
                elif isinstance(v, dict):
                    for idx, (key, value) in enumerate(v.items()):
                        wandb.log(k, {"{}_{}".format(k, key): value}, total_num_steps)
                else:
                    wandb.log({k: v}, step=total_num_steps)
            else:
                if isinstance(v, list):
                    for idx, env_name in enumerate(self.multi_envs):
                        self.writter.add_scalars(k, {"{}_{}".format(k, env_name): v[idx]}, total_num_steps)
                elif isinstance(v, dict):
                    for idx, (key, value) in enumerate(v.items()):
                        self.writter.add_scalars(k, {"{}_{}".format(k, key): value}, total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: v}, total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_obs_tuple, eval_share_obs_tuple, eval_available_actions_tuple, idxs_tuple = self.eval_envs.reset()
        # actully self.n_eval_rollout_threads = 1
        eval_episode_rewards = [[] for _ in idxs_tuple]
        one_episode_rewards = [[] for _ in idxs_tuple]
        eval_battles_won = [0 for _ in idxs_tuple]
        eval_episode = [0 for _ in idxs_tuple]
        eval_done = [False for _ in idxs_tuple]

        eval_rnn_states_list = []
        eval_masks_list = []
        for idx in idxs_tuple:
            eval_rnn_states_list.append(
                np.zeros((self.n_eval_rollout_threads, self.num_agents_list[idx], *self.buffer.buffer_lists[idx].rnn_states.shape[3:]), dtype=np.float32)
            )
            eval_masks_list.append(
                np.ones((self.n_eval_rollout_threads, self.num_agents_list[idx], 1), dtype=np.float32)
            )
        # eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        # eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()

            eval_obs_tuple = [np.concatenate(eval_obs_tuple[idx]) for idx in idxs_tuple]
            eval_rnn_states_list = [np.concatenate(eval_rnn_states_list[idx]) for idx in idxs_tuple]
            eval_masks_list = [np.concatenate(eval_masks_list[idx]) for idx in idxs_tuple]
            eval_available_actions_tuple = [np.concatenate(eval_available_actions_tuple[idx]) for idx in idxs_tuple]
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(
                    eval_obs_tuple, eval_rnn_states_list, eval_masks_list, eval_available_actions_tuple,
                    deterministic=True, n_agents=self.num_agents_list, n_enemies=self.num_enemies_list
                )

            th_na_list = [eval_ob.shape[0] for eval_ob in eval_obs_tuple]
            eval_actions_list = [np.array(np.split(_t2n(item), self.n_eval_rollout_threads)) for item in torch.split(eval_actions, th_na_list, dim=0)]
            
            if self.all_args.use_naive_recurrent_policy or self.all_args.use_recurrent_policy:
                eval_rnn_states_list = [np.array(np.split(_t2n(item), self.n_eval_rollout_threads)) for item in torch.split(eval_rnn_states, th_na_list, dim=0)]
            # elif "rnn_cognition" in self.algorithm_name:
            #     eval_rnn_states_list = [np.array(np.split(_t2n(item), self.n_eval_rollout_threads)) for item in eval_rnn_states]
            else:
                eval_rnn_states_list = [np.array(np.split(_t2n(item), self.n_eval_rollout_threads)) for item in eval_rnn_states]

            # Obser reward and next obs
            eval_obs_tuple, eval_share_obs_tuple, eval_rewards_tuple, \
                eval_dones_tuple, eval_infos_tuple, eval_available_actions_tuple, idxs_tuple = self.eval_envs.step(eval_actions_list)
            
            for eval_rewards, eval_dones, eval_infos, idx in zip(eval_rewards_tuple, eval_dones_tuple, eval_infos_tuple, idxs_tuple):
                one_episode_rewards[idx].append(eval_rewards)

                eval_dones_env = np.all(eval_dones, axis=1)

                eval_rnn_states_list[idx][eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents_list[idx], *self.buffer.buffer_lists[idx].rnn_states.shape[3:]), dtype=np.float32)

                eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents_list[idx], 1), dtype=np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents_list[idx], 1), dtype=np.float32)
                eval_masks_list[idx] = eval_masks
                
                for eval_i in range(self.n_eval_rollout_threads):
                    if eval_dones_env[eval_i]:
                        eval_episode[idx] += 1
                        eval_episode_rewards[idx].append(np.sum(one_episode_rewards[idx], axis=0))
                        one_episode_rewards[idx] = []
                        if eval_infos[eval_i][0]['won']:
                            eval_battles_won[idx] += 1
            
            for idx in idxs_tuple:
                if eval_episode[idx] >= self.all_args.eval_episodes and eval_done[idx] == False:
                    eval_done[idx] = True
                    eval_env_infos = {'eval_average_episode_rewards_{}'.format(self.multi_envs[idx]): np.array(eval_episode_rewards[idx])}                
                    self.log_env(eval_env_infos, total_num_steps, 'eval_average_episode_rewards')
                    eval_win_rate = eval_battles_won[idx]/eval_episode[idx]
                    print("{} eval win rate is {}.".format(self.multi_envs[idx], eval_win_rate))
                    if self.use_wandb:
                        wandb.log({"eval_win_rate_{}".format(self.multi_envs[idx]): eval_win_rate}, step=total_num_steps)
                    else:
                        self.writter.add_scalars("eval_win_rate", {"eval_win_rate_{}".format(self.multi_envs[idx]): eval_win_rate}, total_num_steps)
            
            if np.all(eval_done):
                break

    '''
    Redefine some methods of base_runner.Runner
    '''
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        cent_obs = [np.concatenate(self.buffer.buffer_lists[idx].share_obs[-1]) for idx in range(self.num_multi_envs)]
        rnn_state_critic = [np.concatenate(self.buffer.buffer_lists[idx].rnn_states_critic[-1]) for idx in range(self.num_multi_envs)]
        masks = [np.concatenate(self.buffer.buffer_lists[idx].masks[-1]) for idx in range(self.num_multi_envs)]
        
        th_na_list = [cent_ob.shape[0] for cent_ob in cent_obs]
        next_value = self.trainer.policy.get_values(
            cent_obs, rnn_state_critic, masks,
            n_agents=self.num_agents_list,
            n_enemies=self.num_enemies_list
        )
        next_values = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(next_value, th_na_list, dim=0)]
        
        for idx, next_value in zip(range(self.num_multi_envs), next_values):
            self.buffer.compute_returns(idx, next_value, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        return train_infos


    def log_env(self, env_infos, total_num_steps, main_env_infos=None):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    if main_env_infos is None:
                        self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
                    else:
                        self.writter.add_scalars(main_env_infos, {k: np.mean(v)}, total_num_steps)

    @torch.no_grad()
    def eval4traj(self, total_num_steps):
        eval_obs_tuple, eval_share_obs_tuple, eval_available_actions_tuple, idxs_tuple = self.eval_envs.reset()
        # actully self.n_eval_rollout_threads = 1
        eval_episode_rewards = [[] for _ in idxs_tuple]
        one_episode_rewards = [[] for _ in idxs_tuple]
        eval_battles_won = [0 for _ in idxs_tuple]
        eval_episode = [0 for _ in idxs_tuple]
        eval_done = [False for _ in idxs_tuple]

        eval_rnn_states_list = []
        eval_masks_list = []
        eval_entity_obs_lists = []
        eval_cognition_lists = []
        eval_strategy_lists = []
        eval_done_lists = []
        eval_win_lists = []
        eval_actions_lists = []
        for idx in idxs_tuple:
            eval_rnn_states_list.append(
                np.zeros((self.n_eval_rollout_threads, self.num_agents_list[idx], *self.buffer.buffer_lists[idx].rnn_states.shape[3:]), dtype=np.float32)
            )
            eval_masks_list.append(
                np.ones((self.n_eval_rollout_threads, self.num_agents_list[idx], 1), dtype=np.float32)
            )
        
        # init subtask
        eval_strategies = None

        while True:
            self.trainer.prep_rollout()

            eval_obs_tuple = [np.concatenate(eval_obs_tuple[idx]) for idx in idxs_tuple]
            if eval_strategies is None:
                eval_strategies = [torch.zeros(*eval_obs.shape[:-1], self.all_args.num_subtask, dtype=torch.float32) for eval_obs in eval_obs_tuple]
            eval_strategy_list = [_t2n(item) for item in eval_strategies]
            eval_obs_tuple = [np.concatenate([eval_obs, eval_strategy], axis=-1) for eval_obs, eval_strategy in zip(eval_obs_tuple, eval_strategy_list)]
            
            eval_rnn_states_list = [np.concatenate(eval_rnn_states_list[idx]) for idx in idxs_tuple]
            eval_masks_list = [np.concatenate(eval_masks_list[idx]) for idx in idxs_tuple]
            eval_available_actions_tuple = [np.concatenate(eval_available_actions_tuple[idx]) for idx in idxs_tuple]
            eval_actions, eval_rnn_states, eval_entity_obs, eval_strategy = \
                self.trainer.policy.act(
                    eval_obs_tuple, eval_rnn_states_list, eval_masks_list, eval_available_actions_tuple,
                    deterministic=True, n_agents=self.num_agents_list, n_enemies=self.num_enemies_list
                )

            eval_entity_obs_list = [np.array(np.split(_t2n(item), self.n_eval_rollout_threads)) for item in eval_entity_obs]
            eval_strategy_list = [np.array(np.split(_t2n(item), self.n_eval_rollout_threads)) for item in eval_strategy]
            eval_entity_obs_lists.append(eval_entity_obs_list)
            eval_strategy_lists.append(eval_strategy_list)

            th_na_list = [eval_ob.shape[0] for eval_ob in eval_obs_tuple]
            eval_actions_list = [np.array(np.split(_t2n(item), self.n_eval_rollout_threads)) for item in torch.split(eval_actions, th_na_list, dim=0)]
            eval_actions_lists.append(eval_actions_list)
            if self.all_args.use_naive_recurrent_policy or self.all_args.use_recurrent_policy:
                eval_rnn_states_list = [np.array(np.split(_t2n(item), self.n_eval_rollout_threads)) for item in torch.split(eval_rnn_states, th_na_list, dim=0)]
            # elif "rnn_cognition" in self.algorithm_name:
            #     eval_rnn_states_list = [np.array(np.split(_t2n(item), self.n_eval_rollout_threads)) for item in eval_rnn_states]
            else:
                eval_rnn_states_list = [np.array(np.split(_t2n(item), self.n_eval_rollout_threads)) for item in eval_rnn_states]

            # Obser reward and next obs
            eval_obs_tuple, eval_share_obs_tuple, eval_rewards_tuple, \
                eval_dones_tuple, eval_infos_tuple, eval_available_actions_tuple, idxs_tuple = self.eval_envs.step(eval_actions_list)
            # Done
            eval_done_lists.append([np.all(eval_done, axis=1) for eval_done in eval_dones_tuple])
            # Win
            eval_win_flag = [np.array([False]) for _ in eval_dones_tuple]

            for eval_rewards, eval_dones, eval_infos, idx in zip(eval_rewards_tuple, eval_dones_tuple, eval_infos_tuple, idxs_tuple):
                one_episode_rewards[idx].append(eval_rewards)

                eval_dones_env = np.all(eval_dones, axis=1)

                eval_rnn_states_list[idx][eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents_list[idx], *self.buffer.buffer_lists[idx].rnn_states.shape[3:]), dtype=np.float32)

                eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents_list[idx], 1), dtype=np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents_list[idx], 1), dtype=np.float32)
                eval_masks_list[idx] = eval_masks
                
                for eval_i in range(self.n_eval_rollout_threads):
                    if eval_dones_env[eval_i]:
                        eval_episode[idx] += 1
                        eval_episode_rewards[idx].append(np.sum(one_episode_rewards[idx], axis=0))
                        one_episode_rewards[idx] = []
                        if eval_infos[eval_i][0]['won']:
                            eval_battles_won[idx] += 1
                            eval_win_flag[idx] = np.array([True])
            eval_win_lists.append(eval_win_flag)
            
            for idx in idxs_tuple:
                if eval_episode[idx] >= self.all_args.eval_episodes and eval_done[idx] == False:
                    eval_done[idx] = True
                    eval_env_infos = {'eval_average_episode_rewards_{}'.format(self.multi_envs[idx]): np.array(eval_episode_rewards[idx])}                
                    self.log_env(eval_env_infos, total_num_steps, 'eval_average_episode_rewards')
                    eval_win_rate = eval_battles_won[idx]/eval_episode[idx]
                    print("{} eval win rate is {}.".format(self.multi_envs[idx], eval_win_rate))
                    if self.use_wandb:
                        wandb.log({"eval_win_rate_{}".format(self.multi_envs[idx]): eval_win_rate}, step=total_num_steps)
                    else:
                        self.writter.add_scalars("eval_win_rate", {"eval_win_rate_{}".format(self.multi_envs[idx]): eval_win_rate}, total_num_steps)
            
            if np.all(eval_done):
                self.save_trajectory(eval_entity_obs_lists, eval_strategy_lists, eval_done_lists, eval_win_lists, eval_actions_lists)
                break
    
    @torch.no_grad()
    def eval4attentraj(self, total_num_steps):
        eval_obs_tuple, eval_share_obs_tuple, eval_available_actions_tuple, idxs_tuple = self.eval_envs.reset()
        # actully self.n_eval_rollout_threads = 1
        eval_episode_rewards = [[] for _ in idxs_tuple]
        one_episode_rewards = [[] for _ in idxs_tuple]
        eval_battles_won = [0 for _ in idxs_tuple]
        eval_episode = [0 for _ in idxs_tuple]
        eval_done = [False for _ in idxs_tuple]

        eval_rnn_states_list = []
        eval_masks_list = []
        eval_entity_obs_lists = []
        eval_cognition_lists = []
        eval_strategy_lists = []
        eval_done_lists = []
        eval_win_lists = []
        eval_actions_lists = []
        eval_attentions_lists = []
        for idx in idxs_tuple:
            eval_rnn_states_list.append(
                np.zeros((self.n_eval_rollout_threads, self.num_agents_list[idx], *self.buffer.buffer_lists[idx].rnn_states.shape[3:]), dtype=np.float32)
            )
            eval_masks_list.append(
                np.ones((self.n_eval_rollout_threads, self.num_agents_list[idx], 1), dtype=np.float32)
            )
        
        # init subtask
        eval_strategies = None

        while True:
            self.trainer.prep_rollout()

            eval_obs_tuple = [np.concatenate(eval_obs_tuple[idx]) for idx in idxs_tuple]
            if eval_strategies is None:
                eval_strategies = [torch.zeros(*eval_obs.shape[:-1], self.all_args.num_subtask, dtype=torch.float32) for eval_obs in eval_obs_tuple]
            eval_strategy_list = [_t2n(item) for item in eval_strategies]
            eval_obs_tuple = [np.concatenate([eval_obs, eval_strategy], axis=-1) for eval_obs, eval_strategy in zip(eval_obs_tuple, eval_strategy_list)]
            
            eval_rnn_states_list = [np.concatenate(eval_rnn_states_list[idx]) for idx in idxs_tuple]
            eval_masks_list = [np.concatenate(eval_masks_list[idx]) for idx in idxs_tuple]
            eval_available_actions_tuple = [np.concatenate(eval_available_actions_tuple[idx]) for idx in idxs_tuple]
            eval_actions, eval_rnn_states, eval_entity_obs, eval_strategy, eval_attention = \
                self.trainer.policy.act(
                    eval_obs_tuple, eval_rnn_states_list, eval_masks_list, eval_available_actions_tuple,
                    deterministic=True, n_agents=self.num_agents_list, n_enemies=self.num_enemies_list
                )

            eval_entity_obs_list = [np.array(np.split(_t2n(item), self.n_eval_rollout_threads)) for item in eval_entity_obs]
            eval_strategy_list = [np.array(np.split(_t2n(item), self.n_eval_rollout_threads)) for item in eval_strategy]
            eval_attention_list = [np.array(np.split(_t2n(item), self.n_eval_rollout_threads)) for item in eval_attention]
            eval_entity_obs_lists.append(eval_entity_obs_list)
            eval_strategy_lists.append(eval_strategy_list)
            eval_attentions_lists.append(eval_attention_list)

            th_na_list = [eval_ob.shape[0] for eval_ob in eval_obs_tuple]
            eval_actions_list = [np.array(np.split(_t2n(item), self.n_eval_rollout_threads)) for item in torch.split(eval_actions, th_na_list, dim=0)]
            eval_actions_lists.append(eval_actions_list)
            if self.all_args.use_naive_recurrent_policy or self.all_args.use_recurrent_policy:
                eval_rnn_states_list = [np.array(np.split(_t2n(item), self.n_eval_rollout_threads)) for item in torch.split(eval_rnn_states, th_na_list, dim=0)]
            # elif "rnn_cognition" in self.algorithm_name:
            #     eval_rnn_states_list = [np.array(np.split(_t2n(item), self.n_eval_rollout_threads)) for item in eval_rnn_states]
            else:
                eval_rnn_states_list = [np.array(np.split(_t2n(item), self.n_eval_rollout_threads)) for item in eval_rnn_states]

            # Obser reward and next obs
            eval_obs_tuple, eval_share_obs_tuple, eval_rewards_tuple, \
                eval_dones_tuple, eval_infos_tuple, eval_available_actions_tuple, idxs_tuple = self.eval_envs.step(eval_actions_list)
            # Done
            eval_done_lists.append([np.all(eval_done, axis=1) for eval_done in eval_dones_tuple])
            # Win
            eval_win_flag = [np.array([False]) for _ in eval_dones_tuple]

            for eval_rewards, eval_dones, eval_infos, idx in zip(eval_rewards_tuple, eval_dones_tuple, eval_infos_tuple, idxs_tuple):
                one_episode_rewards[idx].append(eval_rewards)

                eval_dones_env = np.all(eval_dones, axis=1)

                eval_rnn_states_list[idx][eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents_list[idx], *self.buffer.buffer_lists[idx].rnn_states.shape[3:]), dtype=np.float32)

                eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents_list[idx], 1), dtype=np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents_list[idx], 1), dtype=np.float32)
                eval_masks_list[idx] = eval_masks
                
                for eval_i in range(self.n_eval_rollout_threads):
                    if eval_dones_env[eval_i]:
                        eval_episode[idx] += 1
                        eval_episode_rewards[idx].append(np.sum(one_episode_rewards[idx], axis=0))
                        one_episode_rewards[idx] = []
                        if eval_infos[eval_i][0]['won']:
                            eval_battles_won[idx] += 1
                            eval_win_flag[idx] = np.array([True])
            eval_win_lists.append(eval_win_flag)
            
            for idx in idxs_tuple:
                if eval_episode[idx] >= self.all_args.eval_episodes and eval_done[idx] == False:
                    eval_done[idx] = True
                    eval_env_infos = {'eval_average_episode_rewards_{}'.format(self.multi_envs[idx]): np.array(eval_episode_rewards[idx])}                
                    self.log_env(eval_env_infos, total_num_steps, 'eval_average_episode_rewards')
                    eval_win_rate = eval_battles_won[idx]/eval_episode[idx]
                    print("{} eval win rate is {}.".format(self.multi_envs[idx], eval_win_rate))
                    if self.use_wandb:
                        wandb.log({"eval_win_rate_{}".format(self.multi_envs[idx]): eval_win_rate}, step=total_num_steps)
                    else:
                        self.writter.add_scalars("eval_win_rate", {"eval_win_rate_{}".format(self.multi_envs[idx]): eval_win_rate}, total_num_steps)
            
            if np.all(eval_done):
                self.save_trajectory(eval_entity_obs_lists, eval_strategy_lists, eval_done_lists, eval_win_lists, eval_actions_lists, eval_attentions_lists)
                break
    
    def save_trajectory(self, entity_obs_lists, strategy_lists, eval_done_lists, eval_win_lists, eval_actions_lists, eval_attentions_lists=None):
        for idx, entity_ob in enumerate(zip(*entity_obs_lists)):
            entity_obs = np.concatenate(entity_ob, axis=0)
            np.save(os.path.join(self.trajectory_dir, f"{self.multi_envs[idx]}_obs.npy"), arr=entity_obs)
        
        for idx, strategy in enumerate(zip(*strategy_lists)):
            strategies = np.concatenate(strategy, axis=0)
            np.save(os.path.join(self.trajectory_dir, f"{self.multi_envs[idx]}_strategy.npy"), arr=strategies)

        for idx, done in enumerate(zip(*eval_done_lists)):
            dones = np.concatenate(done, axis=0)
            np.save(os.path.join(self.trajectory_dir, f"{self.multi_envs[idx]}_done.npy"), arr=dones)

        for idx, win_flag in enumerate(zip(*eval_win_lists)):
            win_flags = np.concatenate(win_flag, axis=0)
            np.save(os.path.join(self.trajectory_dir, f"{self.multi_envs[idx]}_winflag.npy"), arr=win_flags)

        for idx, eval_action in enumerate(zip(*eval_actions_lists)):
            eval_actions = np.concatenate(eval_action, axis=0)
            np.save(os.path.join(self.trajectory_dir, f"{self.multi_envs[idx]}_actions.npy"), arr=eval_actions)

        if eval_attentions_lists is not None:
            for idx, eval_attention in enumerate(zip(*eval_attentions_lists)):
                eval_attention = np.concatenate(eval_attention, axis=0)
                np.save(os.path.join(self.trajectory_dir, f"{self.multi_envs[idx]}_attention.npy"), arr=eval_attention)

    @torch.no_grad()
    def collect4subtask(self, step):
        self.trainer.prep_rollout()
        # (threads*na, _size)
        cent_obs = [np.concatenate(self.buffer.buffer_lists[idx].share_obs[step]) for idx in range(self.num_multi_envs)]
        obs = [np.concatenate(self.buffer.buffer_lists[idx].obs[step]) for idx in range(self.num_multi_envs)]
        rnn_state = [np.concatenate(self.buffer.buffer_lists[idx].rnn_states[step]) for idx in range(self.num_multi_envs)]
        rnn_state_critic = [np.concatenate(self.buffer.buffer_lists[idx].rnn_states_critic[step]) for idx in range(self.num_multi_envs)]
        masks = [np.concatenate(self.buffer.buffer_lists[idx].masks[step]) for idx in range(self.num_multi_envs)]
        available_actions = [np.concatenate(self.buffer.buffer_lists[idx].available_actions[step]) for idx in range(self.num_multi_envs)]
        th_na_list = [cent_ob.shape[0] for cent_ob in cent_obs]
        
        value, action, action_log_prob, rnn_state, rnn_state_critic, subtasks_actor, subtasks_critic = \
            self.trainer.policy.get_actions(
                cent_obs, obs, rnn_state, rnn_state_critic, masks, available_actions, 
                n_agents=self.num_agents_list, 
                n_enemies=self.num_enemies_list,
            )

        values = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(value, th_na_list, dim=0)]
        actions = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(action, th_na_list, dim=0)]
        action_log_probs = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(action_log_prob, th_na_list, dim=0)]
        # 应该是用不到这个函数
        if self.all_args.use_naive_recurrent_policy or self.all_args.use_recurrent_policy:
            # 多个环境的rnn_state被放到了一个列表中
            rnn_states = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(rnn_state, th_na_list, dim=0)]
            rnn_states_critic = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in torch.split(rnn_state_critic, th_na_list, dim=0)]
        else:
            # rnn_state仍是tensor而不是若干个tensor组成的list
            rnn_states = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in rnn_state]
            rnn_states_critic = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in rnn_state_critic]
            subtasks_actor = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in subtasks_actor]
            subtasks_critic = [np.array(np.split(_t2n(item), self.num_thread_per_env)) for item in subtasks_critic]
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, subtasks_actor, subtasks_critic
    

    def collect_data4subtask(self, num_samples):
        for step in range(num_samples):
            # Sample actions
            values_list, actions_list, action_log_probs_list, \
                rnn_states_list, rnn_states_critic_list, subtasks_actor, subtasks_critic = self.collect4subtask(step)
            # Obser reward and next obs
            obs_tuple, share_obs_tuple, rewards_tuple, dones_tuple, \
                infos_tuple, available_actions_tuple, idxs_tuple = self.envs.step(actions_list)
            
            infos_list = [None for _ in idxs_tuple]
            
            # concat obs/share_obs with subtask
            obs_tuple = [np.concatenate([obs, subtask_a], axis=-1) for obs, subtask_a in zip(obs_tuple, subtasks_actor)]
            # share_obs_tuple = [np.concatenate([share_obs, subtask_c], axis=-1) for share_obs, subtask_c in zip(share_obs_tuple, subtasks_actor)]
            share_obs_tuple = [np.concatenate([share_obs, subtask_c], axis=-1) for share_obs, subtask_c in zip(share_obs_tuple, subtasks_critic)]
            
            for obs, share_obs, rewards, dones, infos, available_actions, \
                values, actions, action_log_probs, rnn_states, rnn_states_critic, idx in zip(
                    obs_tuple, share_obs_tuple, rewards_tuple, dones_tuple, infos_tuple, available_actions_tuple, \
                        values_list, actions_list, action_log_probs_list, rnn_states_list, rnn_states_critic_list, idxs_tuple
                ):
                data = obs, share_obs, rewards, dones, infos, available_actions, \
                    values, actions, action_log_probs, \
                    rnn_states, rnn_states_critic 
                
                # insert data into buffer/enc_buffer
                self.insert(data, idx)
                # infos
                infos_list[idx] = infos

        return infos_list

    def warmup4subtask(self):
        # reset env
        obs_tuple, share_obs_tuple, available_actions_tuple, idxs_tuple = self.envs.reset()

        for obs, share_obs, available_actions, idx in zip(obs_tuple, share_obs_tuple, available_actions_tuple, idxs_tuple):

            # replay buffer
            if not self.use_centralized_V:
                share_obs = obs

            self.buffer.buffer_lists[idx].share_obs[0] = np.concatenate([share_obs, np.zeros([self.num_thread_per_env, self.num_agents_list[idx], self.all_args.num_subtask])], axis=-1).copy()
            self.buffer.buffer_lists[idx].obs[0] = np.concatenate([obs, np.zeros([self.num_thread_per_env, self.num_agents_list[idx], self.all_args.num_subtask])], axis=-1).copy()
            self.buffer.buffer_lists[idx].available_actions[0] = available_actions.copy()


    def evaluate4replay(self):
        if self.all_args.record_attention:
            self.eval4attentraj(total_num_steps=0)
        elif "subtask" in self.algorithm_name:
            self.eval4traj(total_num_steps=0)
        else:
            self.eval(total_num_steps=0)
        if self.all_args.save_replay:
            self.eval_envs.save_replay()
    # def save(self):
    #     super().save()
    #     policy_context_encoder = self.trainer.policy.context_encoder
    #     torch.save(policy_context_encoder.state_dict(), str(self.save_dir) + "/context_encoder.pt")
    
    # def restore(self):
    #     super().restore()
    #     policy_context_encoder_state_dict = torch.load(str(self.model_dir) + "/context_encoder.pt")
    #     self.policy.context_encoder.load_state_dict(policy_context_encoder_state_dict)

    # def restore_only_con_encoder(self):
    #     '''
    #     just restore context encoder when transfer from a pearl mappo model
    #     '''
    #     policy_context_encoder_state_dict = torch.load(str(self.model_dir) + "/context_encoder.pt")
    #     self.policy.context_encoder.load_state_dict(policy_context_encoder_state_dict)