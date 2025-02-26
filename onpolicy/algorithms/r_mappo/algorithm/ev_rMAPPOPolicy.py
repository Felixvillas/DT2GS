import torch
import torch.nn.functional as F
import numpy as np

from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.algorithms.r_mappo.algorithm.ev_actor_critic import R_Actor, R_Critic


def add_list(*args):
        return ([arg] for arg in args)

class EntityVAER_MAPPOPolicy(R_MAPPOPolicy):
    def __init__(
        self, 
        args, 
        multi_envs, 
        num_thread_per_env, 
        num_agents_list, 
        num_enemies_list,
        obs_space, 
        cent_obs_space, 
        act_space, 
        device=torch.device('cpu')
    ):
        print("-------------------------------------------------------------------------------------------")
        print("====================----------Entity VAE Multi R_MAPPO Policy---------=====================")
        print("-------------------------------------------------------------------------------------------")
        self.args = args
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device, num_agents_list, num_enemies_list)
        self.critic = R_Critic(args, self.share_obs_space, self.device, num_agents_list, num_enemies_list)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

        self.tpdv = dict(dtype=torch.float32, device=device)

    '''
    functions need to rewrite
    '''
    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False, n_agents=None, n_enemies=None, is_loss=False):
        if "subtask" in self.args.algorithm_name:
            actions, action_log_probs, rnn_states_actor, _, subtasks_actor = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic,
                                                                 n_agents,
                                                                 n_enemies,
                                                                 is_loss=False)
            values, rnn_states_critic, _, _, subtasks_critic = self.critic(cent_obs, rnn_states_critic, masks, n_agents, n_enemies, is_loss=False)
            return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, subtasks_actor, subtasks_critic
        else:
            actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                    rnn_states_actor,
                                                                    masks,
                                                                    available_actions,
                                                                    deterministic,
                                                                    n_agents,
                                                                    n_enemies,
                                                                    is_loss=False)

            values, rnn_states_critic, _, _ = self.critic(cent_obs, rnn_states_critic, masks, n_agents, n_enemies, is_loss=False)
            return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks, n_agents=None, n_enemies=None, is_loss=False):
        if "subtask" in self.args.algorithm_name:
            values, _, _, _, _ = self.critic(cent_obs, rnn_states_critic, masks, n_agents, n_enemies, is_loss=False)
        else:
            values, _, _, _ = self.critic(cent_obs, rnn_states_critic, masks, n_agents, n_enemies, is_loss=False)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None, n_agents=None, n_enemies=None, is_loss=True):
        action_log_probs, dist_entropy, actor_vae_loss_kl_list, actor_vae_loss_re_list = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks,
                                                                     n_agents,
                                                                     n_enemies,
                                                                     is_loss=True)
        if "subtask" in self.args.algorithm_name:
            values, _, critic_vae_loss_kl_list, critic_vae_loss_re_list, _ = self.critic(cent_obs, rnn_states_critic, masks, n_agents, n_enemies, is_loss=True)
        else:
            values, _, critic_vae_loss_kl_list, critic_vae_loss_re_list = self.critic(cent_obs, rnn_states_critic, masks, n_agents, n_enemies, is_loss=True)
        return values, action_log_probs, dist_entropy, actor_vae_loss_kl_list, actor_vae_loss_re_list, critic_vae_loss_kl_list, critic_vae_loss_re_list

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False, n_agents=None, n_enemies=None, is_loss=False):
        if self.args.record_attention:
            actions, _, rnn_states_actor, entity_obs, strategy, attention = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic, n_agents, n_enemies, is_loss=False)
            return actions, rnn_states_actor, entity_obs, strategy, attention
        elif self.args.algorithm_name == "entity_product" or self.args.algorithm_name == "entity_product_transfer" or \
                self.args.algorithm_name == "subtask" or self.args.algorithm_name == "subtask_transfer":
            actions, _, rnn_states_actor, entity_obs, strategy = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic, n_agents, n_enemies, is_loss=False)
            return actions, rnn_states_actor, entity_obs, strategy
        else:
            actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic, n_agents, n_enemies, is_loss=False)
            return actions, rnn_states_actor

