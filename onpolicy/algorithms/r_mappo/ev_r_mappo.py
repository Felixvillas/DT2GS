import torch
import torch.nn as nn
import numpy as np

from onpolicy.utils.util import get_gard_norm
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.algorithms.r_mappo.algorithm.ev_actor_critic import check

class EntityVAER_MAPPO(R_MAPPO):
    def __init__(self, args, policy, n_agents_list, n_enemies_list, device=torch.device("cpu")):
        print("-------------------------------------------------------------------------------------------")
        print("======================--------------Entity VAE R_MAPPO-------------========================")
        print("-------------------------------------------------------------------------------------------")
        super().__init__(args, policy, device)
        self.args = args
        self.multi_envs = args.multi_envs.split('|')
        self.num_multi_envs = len(self.multi_envs)
        self.n_agents_list = n_agents_list
        self.n_enemies_list = n_enemies_list
    
    def ppo_update(self, sample, update_actor=True):

        value_loss_list, actor_loss_list, dist_entropy, imp_weights, \
            actor_vae_loss_kl_list, actor_vae_loss_re_list, critic_vae_loss_kl_list, critic_vae_loss_re_list \
             = self.get_ppo_loss(sample)
        
        actor_loss = torch.stack(actor_loss_list).mean() + torch.stack(actor_vae_loss_kl_list).mean() + torch.stack(actor_vae_loss_re_list).mean()
        critic_loss = torch.stack(value_loss_list).mean() + torch.stack(critic_vae_loss_kl_list).mean() + torch.stack(critic_vae_loss_re_list).mean()

        # update Actor
        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            actor_loss.backward(retain_graph=True) if self.args.use_actor_loss else actor_loss.backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # update Critic
        self.policy.critic_optimizer.zero_grad()

        critic_loss.backward(retain_graph=True)

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss_list, critic_grad_norm, actor_loss_list, dist_entropy, \
            actor_grad_norm, imp_weights, actor_vae_loss_kl_list, actor_vae_loss_re_list, \
                critic_vae_loss_kl_list, critic_vae_loss_re_list

    
    def cat_sample(self, sample):
        '''
        sample: [[obs1, act1, ...], [obs2, act2], ...],
        in which elem: [obsi, acti] is corresponds to ith env in multi_envs
        '''
        for item in zip(*sample):
            # every item is a tuple of obs/actions/..., in which every elem in this item 
            # corresponding to every multi_env respectively
            yield item

    def get_ppo_loss(self, sample):
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = tuple(self.cat_sample(sample))

        old_action_log_probs_batch = check(old_action_log_probs_batch, self.tpdv)
        bs_na_list = [oalpb.size(0) for oalpb in old_action_log_probs_batch]

        old_action_log_probs_batch = torch.cat(old_action_log_probs_batch)
        adv_targ = torch.cat(check(adv_targ, self.tpdv))
        value_preds_batch = check(value_preds_batch, self.tpdv)
        return_batch = check(return_batch, self.tpdv)

        # Reshape to do in a single forward pass for all steps
        # values: (bs_na1+bs_na2+..., 1)
        # action_log_probs: (bs_na1+bs_na2+..., 1)
        # dist_entropy: a scalar
        values, action_log_probs, dist_entropy, \
            actor_vae_loss_kl_list, actor_vae_loss_re_list, \
                critic_vae_loss_kl_list, critic_vae_loss_re_list =\
                                                self.policy.evaluate_actions(share_obs_batch,
                                                obs_batch, 
                                                rnn_states_batch, 
                                                rnn_states_critic_batch, 
                                                actions_batch, 
                                                masks_batch, 
                                                available_actions_batch,
                                                active_masks_batch, 
                                                self.n_agents_list,
                                                self.n_enemies_list, 
                                                is_loss=True)
        active_masks_batch = check(active_masks_batch, self.tpdv)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        surr1_list = torch.split(surr1, bs_na_list, dim=0)
        surr2_list = torch.split(surr2, bs_na_list, dim=0)
        value_list = torch.split(values, bs_na_list, dim=0)

        value_loss_list = []
        actor_loss_list = []
        
        for surr1_i, surr2_i, active_masks_batch_i, dist_entropy_i, \
            value_i, value_preds_batch_i, return_batch_i in zip(
            surr1_list, surr2_list, active_masks_batch, dist_entropy, value_list,
            value_preds_batch, return_batch
        ):

            if self._use_policy_active_masks:
                policy_action_loss = (-torch.sum(torch.min(surr1_i, surr2_i),
                                                dim=-1,
                                                keepdim=True) * active_masks_batch_i).sum() / active_masks_batch_i.sum()
            else:
                policy_action_loss = -torch.sum(torch.min(surr1_i, surr2_i), dim=-1, keepdim=True).mean()

            policy_loss = policy_action_loss
            actor_loss = policy_loss - dist_entropy_i * self.entropy_coef

            # critic update
            value_loss = self.cal_value_loss(value_i, value_preds_batch_i, return_batch_i, active_masks_batch_i) * self.value_loss_coef
            
            value_loss_list.append(value_loss)
            actor_loss_list.append(actor_loss)
        return value_loss_list, actor_loss_list, dist_entropy, imp_weights, actor_vae_loss_kl_list, actor_vae_loss_re_list, critic_vae_loss_kl_list, critic_vae_loss_re_list

    def train(self, buffer, update_actor=True):
        self.multi_actor_loss, self.multi_critic_loss = None, None
        multi_advantages = []
        for idx in range(self.num_multi_envs):
            if self._use_popart or self._use_valuenorm:
                advantages = buffer.buffer_lists[idx].returns[:-1] - self.value_normalizer.denormalize(buffer.buffer_lists[idx].value_preds[:-1])
            else:
                advantages = buffer.buffer_lists[idx].returns[:-1] - buffer.buffer_lists[idx].value_preds[:-1]
            advantages_copy = advantages.copy()
            advantages_copy[buffer.buffer_lists[idx].active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
            multi_advantages.append(advantages)
        
        train_info = {}

        train_info['value_loss'] = dict(zip(self.multi_envs, [0 for _ in self.multi_envs]))
        train_info['policy_loss'] = dict(zip(self.multi_envs, [0 for _ in self.multi_envs]))
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['actor_kl_loss'] = dict(zip(self.multi_envs, [0 for _ in self.multi_envs]))
        train_info['actor_re_loss'] = dict(zip(self.multi_envs, [0 for _ in self.multi_envs]))
        train_info['critic_kl_loss'] = dict(zip(self.multi_envs, [0 for _ in self.multi_envs]))
        train_info['critic_re_loss'] = dict(zip(self.multi_envs, [0 for _ in self.multi_envs]))

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy or "rnn_cognition" in self.args.algorithm_name or "subtask" in self.args.algorithm_name:
                # print("1, recurrent_generator")
                data_generator = buffer.recurrent_generator(multi_advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                # print("2, naive_recurrent_generator")
                data_generator = buffer.naive_recurrent_generator(multi_advantages, self.num_mini_batch)
            else:
                # print("3, feed_forward_generator")
                data_generator = buffer.feed_forward_generator(multi_advantages, self.num_mini_batch)
                # data_generator = buffer.new_feed_forward_generator(multi_advantages, self.num_mini_batch)
                # data_generator = buffer.entity_feed_forward_generator(multi_advantages, self.num_mini_batch)

            for idx, sample in enumerate(zip(*data_generator)):

                value_loss_list, critic_grad_norm, actor_loss_list, dist_entropy, actor_grad_norm, \
                    imp_weights, actor_vae_loss_kl_list, actor_vae_loss_re_list, \
                        critic_vae_loss_kl_list, critic_vae_loss_re_list \
                        = self.ppo_update(sample, update_actor)

                for idx, value_loss in enumerate(value_loss_list):
                    train_info['value_loss'][self.multi_envs[idx]] \
                        += value_loss.item()
                for idx, actor_loss in enumerate(actor_loss_list):
                    train_info['policy_loss'][self.multi_envs[idx]] \
                        += actor_loss.item()
                train_info['dist_entropy'] += torch.stack(dist_entropy).mean().item()
                train_info['actor_grad_norm'] += actor_grad_norm.item()
                train_info['critic_grad_norm'] += critic_grad_norm.item()
                train_info['ratio'] += imp_weights.mean().item()
                for idx, actor_kl_loss in enumerate(actor_vae_loss_kl_list):
                    # print(actor_kl_loss)
                    train_info['actor_kl_loss'][self.multi_envs[idx]] \
                        += actor_kl_loss.item()
                for idx, actor_re_loss in enumerate(actor_vae_loss_re_list):
                    train_info['actor_re_loss'][self.multi_envs[idx]] \
                        += actor_re_loss.item()
                for idx, critic_kl_loss in enumerate(critic_vae_loss_kl_list):
                    train_info['critic_kl_loss'][self.multi_envs[idx]] \
                        += critic_kl_loss.item()
                for idx, critic_re_loss in enumerate(critic_vae_loss_re_list):
                    train_info['critic_re_loss'][self.multi_envs[idx]] \
                        += critic_re_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            if isinstance(train_info[k], dict):
                for key in train_info[k].keys():
                    train_info[k][key] /= num_updates
            else:
                train_info[k] /= num_updates

        return train_info

