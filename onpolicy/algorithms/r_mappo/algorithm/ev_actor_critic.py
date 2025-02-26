import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from onpolicy.algorithms.utils.util import init
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.algorithms.utils.distributions import FixedCategorical
from torch.distributions import Categorical



def check(inputs, device):
    '''
    input is numpy/tensor or a list of numpy/tensor
    device is a dict consist of dtype and device of torch.device
    output is a list of tensor
    '''
    if isinstance(inputs, list) or isinstance(inputs, tuple):
        output = [torch.from_numpy(input).to(**device) if type(input) == np.ndarray else input.to(**device) for input in inputs]
    else:
        output = torch.from_numpy(inputs).to(**device) if type(inputs) == np.ndarray else inputs.to(**device)
    return output

class EntityVAECategorical(nn.Module):
    def __init__(self, args, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(EntityVAECategorical, self).__init__()
        self.args = args
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))
    
    def forward_i(self, x, available_actions=None, n_enemies=None):
        action_value = self.linear(torch.cat([x[:, :1], x[:, -n_enemies:]], dim=-2))
        x_basic = action_value[:, 0]
        x_attack = torch.mean(action_value[:, 1:], dim=-1)
        x = torch.cat([x_basic, x_attack], dim=-1)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return x

    def forward(self, x, available_actions=None, n_enemies=None):
        x = [self.forward_i(x_i, available_actions_i, n_enemy) for x_i, available_actions_i, n_enemy in zip(x, available_actions, n_enemies)]
        return [FixedCategorical(logits=x_i) for x_i in x]

    

class EntityVAEACTLayer(nn.Module):
    def __init__(self, args, action_space, inputs_dim, use_orthogonal, gain):
        super(EntityVAEACTLayer, self).__init__()
        self.args = args
        action_dim = 6

        self.action_out = EntityVAECategorical(args, inputs_dim, action_dim, use_orthogonal, gain)

    def mode(self, action_logits):
        '''
        action_logits is a list of class: FixedCategorical
        '''
        return torch.cat([action_logit.mode() for action_logit in action_logits])

    def sample(self, action_logits):
        '''
        action_logits is a list of class: FixedCategorical
        '''
        return torch.cat([action_logit.sample() for action_logit in action_logits])
    
    def log_probs(self, action_logits, actions, bs_na_list=None):
        '''
        action_logits is a list of class: FixedCategorical
        actions is a list of tensor(action)
        '''
        action_list = torch.split(actions, bs_na_list, dim=0) if bs_na_list is not None else actions
        return torch.cat([action_logit.log_probs(action) for action_logit, action in zip(action_logits, action_list)])

    def dist_entropy_(self, action_logits, active_masks):
        '''
        action_logits is a list of class: FixedCategorical
        active_masks is a list of tensor(action_masks)
        '''
        dist_entropy = []
        if active_masks is not None:
            for action_logit, active_mask in zip(action_logits, active_masks):
                dist_entropy.append(
                    (action_logit.entropy()*active_mask.squeeze(-1)).sum() / active_mask.sum()
                )
            # dist_entropy = \
            #     torch.cat([action_logit.entropy()*active_mask.squeeze(-1) for action_logit, active_mask in zip(action_logits, active_masks)]).sum() /\
            #         torch.cat(active_masks).sum()
        else:
            dist_entropy.extend(
                [action_logit.entropy().mean() for action_logit in action_logits]
            )
            # dist_entropy = torch.cat([action_logit.entropy() for action_logit in action_logits]).mean()
        return dist_entropy

    def forward(self, x, available_actions=None, deterministic=False, n_enemies=None):
        bs_na_list = [x_i.size(0) for x_i in x]
        action_logits = self.action_out(x, available_actions, n_enemies)
        actions = self.mode(action_logits) if deterministic else self.sample(action_logits) 
        action_log_probs = self.log_probs(action_logits, actions, bs_na_list)
        
        return actions, action_log_probs
    
    def forward4subtask(self, x, available_actions=None, deterministic=False, n_enemies=None):
        bs_na_list = [x_i.size(0) for x_i in x]
        action_logits = self.action_out(x, available_actions, n_enemies)
        actions = self.mode(action_logits) if deterministic else self.sample(action_logits) 
        action_log_probs = self.log_probs(action_logits, actions, bs_na_list)
        
        return actions, action_log_probs, action_logits

    def get_probs(self, x, available_actions=None, n_enemies=None):
        raise NotImplementedError
        action_logits = self.action_out(x, available_actions, n_enemies)
        action_probs = action_logits.probs
        
        return action_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None, n_enemies=None):
        action_logits = self.action_out(x, available_actions, n_enemies)
        action_log_probs = self.log_probs(action_logits, action)
        dist_entropy = self.dist_entropy_(action_logits, active_masks)
        
        return action_log_probs, dist_entropy
        

class SelfAttention(nn.Module):
    def __init__(self, args, emb_dim=32, heads=4, use_orthogonal=True) -> None:
        # multi self attention(when heads>1)
        print("----------------------------------------------------------------------------------")
        print("=======================Entity  VAE Use Common SelfAttention=======================")
        print("----------------------------------------------------------------------------------")
        super(SelfAttention, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), args.gain)
        # q, k, v is same at the dim
        self.emb_dim = emb_dim
        self.heads = heads
        self.sqrt_emb_dim = self.emb_dim ** 0.5
        self.q_w = init_(nn.Linear(self.emb_dim * self.heads, self.emb_dim * self.heads))
        self.k_w = init_(nn.Linear(self.emb_dim * self.heads, self.emb_dim * self.heads))
        self.v_w = init_(nn.Linear(self.emb_dim * self.heads, self.emb_dim * self.heads))
        # self.head2emb = init_(nn.Linear(self.emb_dim * self.heads, self.emb_dim))
        # self.talking_w = init_(nn.Linear(args.n_agents+args.n_enemies, args.n_agents+args.n_enemies))
    
    def forward_i(self, features):
        # bs: batch_size, ne: num_eneities, fd: feature_dim, hs: heads
        bs, ne, _ = features.size()
        hs, fd = self.heads, self.emb_dim
        q_wave = self.q_w(features).view(bs, ne, hs, fd)
        k_wave = self.k_w(features).view(bs, ne, hs, fd)
        v_wave = self.v_w(features).view(bs, ne, hs, fd)

        q_wave = q_wave.transpose(1, 2).contiguous().view(bs*hs, ne, fd)
        k_wave = k_wave.transpose(1, 2).contiguous().view(bs*hs, ne, fd)
        v_wave = v_wave.transpose(1, 2).contiguous().view(bs*hs, ne, fd)

        # dot = F.softmax(self.talking_w(torch.matmul(q_wave, k_wave.transpose(-1, -2))/self.sqrt_emb_dim), dim=-1)
        dot = F.softmax(torch.matmul(q_wave, k_wave.transpose(-1, -2))/self.sqrt_emb_dim, dim=-1)
        features_attention = torch.matmul(dot, v_wave).view(bs, hs, ne, fd)
        features_attention = features_attention.transpose(1, 2).contiguous().view(bs, ne, hs*fd)
        # features_attention = self.head2emb(features_attention) 
        return features_attention

    def forward(self, features):
        hs, fd = self.heads, self.emb_dim
        features_attention = torch.cat([self.forward_i(feature).view(-1, hs*fd) for feature in features], dim=0)
        return features_attention

    def forward_a(self, features):
        features_attention = [self.forward_i(feature) for feature in features]
        return features_attention

    def forward_4_adaptive_semantics_i(self, x_emb, subtask_emb):
        # bs: batch_size, ne: num_eneities, fd: feature_dim, hs: heads
        bs, ne, _ = x_emb.size()
        hs, fd = self.heads, self.emb_dim
        q_subtask = self.q_w(subtask_emb).unsqueeze(1).view(bs, 1, hs, fd)
        q_wave = self.q_w(x_emb).view(bs, ne, hs, fd)
        k_wave = self.k_w(x_emb).view(bs, ne, hs, fd)
        v_wave = self.v_w(x_emb).view(bs, ne, hs, fd)

        q_subtask = q_subtask.transpose(1, 2).contiguous().view(bs*hs, 1, fd)
        q_wave = q_wave.transpose(1, 2).contiguous().view(bs*hs, ne, fd)
        k_wave = k_wave.transpose(1, 2).contiguous().view(bs*hs, ne, fd)
        v_wave = v_wave.transpose(1, 2).contiguous().view(bs*hs, ne, fd)

        subtask_dot = F.softmax(torch.matmul(q_subtask, k_wave.transpose(-1, -2)) / self.sqrt_emb_dim, dim=-1)
        subtask_semantics = torch.matmul(subtask_dot, v_wave).view(bs, hs, 1, fd)
        subtask_semantics = subtask_semantics.transpose(1, 2).contiguous().squeeze(1).view(bs, hs*fd)

        dot = F.softmax(torch.matmul(q_wave, k_wave.transpose(-1, -2)) / self.sqrt_emb_dim, dim=-1)
        x_attention = torch.matmul(dot, v_wave).view(bs, hs, ne, fd)
        x_attention = x_attention.transpose(1, 2).contiguous().view(bs, ne, hs*fd)
        # features_attention = self.head2emb(features_attention) 
        return x_attention, subtask_semantics, subtask_dot
    
    def forward_4_adaptive_semantics(self, x_emb, subtask_emb):
        x_emb_tuple, subtask_emb_tuple, subtask_dot_tuple = zip(*[self.forward_4_adaptive_semantics_i(i_x, i_s) for i_x, i_s in zip(x_emb, subtask_emb)])
        x_emb, subtask_emb, subtask_dot = list(x_emb_tuple), list(subtask_emb_tuple), list(subtask_dot_tuple)
        return x_emb, subtask_emb, subtask_dot


class TransformerBlock(nn.Module):
    def __init__(self, args, emb_dim, heads, use_orthogonal, ff_hidden_mult=4, dropout=0.0) -> None:
        super(TransformerBlock, self).__init__()
        self.emb_dim = emb_dim
        self.heads = heads
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), args.gain)
        
        self.attention = SelfAttention(args, emb_dim, heads, use_orthogonal)

        self.norm1 = nn.LayerNorm(emb_dim*heads)
        self.norm2 = nn.LayerNorm(emb_dim*heads)
        
        self.ff = nn.Sequential(
            init_(nn.Linear(emb_dim*heads, ff_hidden_mult*emb_dim)),
            nn.ReLU(),
            init_(nn.Linear(ff_hidden_mult*emb_dim, emb_dim*heads))
        )

        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention(x)
        # attended, _ = self.attention(x, x, x)
        '''
        bs, _, embheads = attended.size()
        pad_tensor = torch.zeros((bs, self.n_agents-1, embheads), dtype=torch.float32, device=attended.device)
        attended = torch.cat([attended[:, :1], pad_tensor, attended[:, 1:]], dim=-2)
        '''
        x = torch.cat([x_i.view(-1, self.emb_dim*self.heads) for x_i in x])
        x = self.norm1(attended + x)
        # x = self.dropout(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)
        # x = self.dropout(x)
        return x

    def forward_4_adaptive_semantics(self, x_emb, subtask_emb):
        """
            forward for adaptive subtask semantics
        """
        x_emb_attended, subtask_emb_attended, subtask_dot = self.attention.forward_4_adaptive_semantics(x_emb, subtask_emb)
        x_emb = [self.norm1(i_x_a + i_x) for i_x, i_x_a in zip(x_emb, x_emb_attended)]
        x_fedforward = [self.ff(i_x) for i_x in x_emb]
        x_emb = [self.norm2(i_xf + i_x) for i_xf, i_x in zip(x_fedforward, x_emb)]
        subtask_emb = [self.norm1(i_s_a + i_s) for i_s, i_s_a in zip(subtask_emb, subtask_emb_attended)]
        s_fedforward = [self.ff(i_s) for i_s in subtask_emb]
        subtask_emb = [self.norm2(i_sf + i_s) for i_sf, i_s in zip(s_fedforward, subtask_emb)]
        return x_emb, subtask_emb, subtask_dot


class Transformer(nn.Module):
    def __init__(self, args, input_dim, latent_dim, emb_dim, heads, depth, output_dim, use_orthogonal, device) -> None:
        super(Transformer, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), args.gain)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.heads = heads
        self.context_encoder = Context_Encoder(args, input_dim, latent_dim, device)
        self.input_embedding = init_(nn.Linear(input_dim + latent_dim, emb_dim*heads))

        tblocks = []
        for _ in range(depth):
            tblocks.append(
                TransformerBlock(args, emb_dim, heads, use_orthogonal)
            )
        self.tblocks = nn.Sequential(*tblocks)
        self.toprobs = init_(nn.Linear(emb_dim*heads, output_dim))
        self.tpdv = dict(dtype=torch.float32, device=device)
    
    def _encode(self, obs, n_agents, n_enemies):
        '''
        obs: a list of ob, in which ob's shape is (bs_na, obs_dim)
        '''
        feat_dim = self.input_dim
        bs_na_list = [ob.size(0) for ob in obs]
        
        split_lists = [[(n_agent+n_enemy)*feat_dim, 4] for n_agent, n_enemy in zip(n_agents, n_enemies)]
        entity_ob_list = [
            torch.split(ob, split_list, dim=-1)[0].contiguous().view(bs_na, n_agent+n_enemy, feat_dim).view(-1, feat_dim) \
                for ob, split_list, bs_na, n_agent, n_enemy in \
                    zip(obs, split_lists, bs_na_list, n_agents, n_enemies)
        ]
        entity_obs = torch.cat(entity_ob_list)
        return entity_obs, bs_na_list

    def _decode(self, entity_obs, n_agents, n_enemies, bs_na_list, feat_dim):
        '''
        obs is a tensor which shape is (-1, feat_dim)
        '''
        split_list = [bs_na * (n_agent + n_enemy) for bs_na, n_agent, n_enemy in zip(bs_na_list, n_agents, n_enemies)]
        entity_ob_list = torch.split(entity_obs, split_list, dim=0)
        entity_ob_list = [
            entity_ob.contiguous().view(bs_na, n_agent+n_enemy, feat_dim) \
                for entity_ob, bs_na, n_agent, n_enemy in zip(entity_ob_list, bs_na_list, n_agents, n_enemies)
        ]
        return entity_ob_list


    def forward(self, x, n_agents, n_enemies, is_loss):
        entity_obs, bs_na_list = self._encode(x, n_agents, n_enemies)
        vae_entity_obs, vae_loss_kl, vae_loss_re = self.context_encoder.encode(entity_obs, is_loss)
        x = self.input_embedding(vae_entity_obs)
        entity_ob_list = self._decode(x, n_agents, n_enemies, bs_na_list, self.emb_dim*self.heads)
        tb = self.tblocks(entity_ob_list)
        x = self.toprobs(tb)
        x_list = self._decode(x, n_agents, n_enemies, bs_na_list, self.output_dim)

        # vae loss
        loss_list = [ib*(ina + ine) for ib, ina, ine in zip(bs_na_list, n_agents, n_enemies)]
        vae_loss_kl_list = torch.split(vae_loss_kl, loss_list, dim=0) if is_loss else None
        vae_loss_re_list = torch.split(vae_loss_re, loss_list, dim=0) if is_loss else None
        vae_loss_kl_list = [i_v_k.mean() for i_v_k in vae_loss_kl_list] if is_loss else None
        # vae_loss_kl_list = [torch.tensor(0.).to(**self.tpdv) for _ in n_agents] if is_loss else None
        vae_loss_re_list = [i_v_r.mean() for i_v_r in vae_loss_re_list] if is_loss else None
        
        return x_list, vae_loss_kl_list, vae_loss_re_list


class R_Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu"), n_agents_list=None, n_enemies_list=None):
        super(R_Actor, self).__init__()
        self.args = args
        self.use_actor_loss = self.args.use_actor_loss
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.actor_feat_dim = 16
        
        if self.args.algorithm_name == "entity" or self.args.algorithm_name == "entity_transfer":
            print("Actor: without strategy")
            self.transformer = Eransformer(
                args, self.actor_feat_dim, self.args.latent_dim, self.hidden_size, 
                args.transformer_heads, args.transformer_depth, 
                self.hidden_size, args.transformer_orth, device
            )
        elif self.args.algorithm_name == "asn":
            print("Actor: ASN")
            self.transformer = ASN(
                args, n_agents_list, n_enemies_list, self.actor_feat_dim, 
                self.args.latent_dim, self.hidden_size, 
                args.transformer_heads, args.transformer_depth, 
                self.hidden_size, args.transformer_orth, device
            )
        elif self.args.algorithm_name == "subtask" or self.args.algorithm_name == "subtask_transfer":
            print("Actor: Construct subtask using cognition in a permutation invariance way")
            self.transformer = SubtaskTransformer(
                args, self.actor_feat_dim, self.args.latent_dim, self.hidden_size, 
                args.transformer_heads, args.transformer_depth, 
                self.hidden_size, args.transformer_orth, device
            )
        elif self.args.algorithm_name == "asn_gatten" or self.args.algorithm_name == "asn_gatten_transfer":
            print("Actor: ASN_Generalization_Attention")
            self.transformer = ASN_G_Atten(
                args, n_agents_list, n_enemies_list, self.actor_feat_dim, 
                self.args.latent_dim, self.hidden_size, 
                args.transformer_heads, args.transformer_depth, 
                self.hidden_size, args.transformer_orth, device
            )
        else:
            raise NotImplementedError

        self.act = EntityVAEACTLayer(args, action_space, self.hidden_size, self._use_orthogonal, self._gain)
        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False, n_agents=None, n_enemies=None, is_loss=False):
        obs = check(obs, self.tpdv)
        rnn_states = check(rnn_states, self.tpdv)
        masks = check(masks, self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions, self.tpdv)
        
        if self.args.record_attention:
            actor_feature_list, rnn_states, _, _, entity_ob_list, subtask_list, subtask_dot = self.transformer(
                obs, rnn_states, masks, 
                n_agents, n_enemies,
                is_loss, collect_record=True, deterministic=deterministic
            )

        elif "subtask" in self.args.algorithm_name:
            # for trajectory record then analysis
            actor_feature_list, rnn_states, _, _, entity_ob_list, subtask_list = self.transformer(
                obs, rnn_states, masks, 
                n_agents, n_enemies,
                is_loss, collect_record=True, deterministic=deterministic
            )

        elif deterministic and (
            self.args.algorithm_name == "entity_product" or self.args.algorithm_name == "entity_product_transfer"
        ):
            # for trajectory record then analysis
            actor_feature_list, rnn_states, _, _, entity_ob_list, subtask_list = self.transformer(
                obs, rnn_states, masks, 
                n_agents, n_enemies,
                is_loss, evaltraj_record=True
            )

        else:
            actor_feature_list, rnn_states, _, _ = self.transformer(
                obs, rnn_states, masks, 
                n_agents, n_enemies,
                is_loss
            )
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            raise NotImplementedError
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # if False:
        #     actions, action_log_probs, action_logits= self.act.forward4subtask(actor_feature_list, available_actions, deterministic, n_enemies)
        #     return actions, action_log_probs, rnn_states, entity_ob_list, subtask_list, action_logits
        # else:
        #     actions, action_log_probs = self.act(actor_feature_list, available_actions, deterministic, n_enemies)
        actions, action_log_probs = self.act(actor_feature_list, available_actions, deterministic, n_enemies)
        if self.args.record_attention:
            return actions, action_log_probs, rnn_states, entity_ob_list, subtask_list, subtask_dot
        # actions: (bs_na1 + bs_na2 + ..., 1)
        # action_log_probs: (bs_na1 + bs_na2 + ..., 1)
        # rnn_states: a list of tensor
        if "subtask" in self.args.algorithm_name or (
                deterministic and (
                self.args.algorithm_name == "entity_product" or self.args.algorithm_name == "entity_product_transfer"
            ) 
        ):
            return actions, action_log_probs, rnn_states, entity_ob_list, subtask_list
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None, n_agents=None, n_enemies=None, is_loss=False):
        obs = check(obs, self.tpdv)
        rnn_states = check(rnn_states, self.tpdv)
        action = check(action, self.tpdv)
        masks = check(masks, self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions, self.tpdv)
        
        if active_masks is not None:
            active_masks = check(active_masks, self.tpdv)

        actor_feature_list, _, vae_loss_kl, vae_loss_re = self.transformer(
            obs, rnn_states, masks, 
            n_agents, n_enemies,
            is_loss
        )
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            raise NotImplementedError
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_feature_list,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None,
                                                                   n_enemies=n_enemies)
        # action_log_probs: (bs_na1 + bs_na2 + ..., 1)
        # dist_entropy: a scalar
        return action_log_probs, dist_entropy, vae_loss_kl, vae_loss_re

class R_Critic(nn.Module):
    def __init__(self, args, cent_obs_space, device=torch.device("cpu"), n_agents_list=None, n_enemies_list=None):
        super(R_Critic, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        if args.use_obs_instead_of_state:
            self.critic_feat_dim = 16
        else:
            self.critic_feat_dim = 20

        
        if self.args.algorithm_name == "entity" or self.args.algorithm_name == "entity_transfer":
            print("Critic: without strategy")
            self.transformer = Eransformer(
                args, self.critic_feat_dim, self.args.latent_dim, self.hidden_size, 
                args.transformer_heads, args.transformer_depth, 
                self.hidden_size, args.transformer_orth, device
            )
        elif self.args.algorithm_name == "asn":
            print("Critic: ASN")
            self.transformer = ASN(
                args, n_agents_list, n_enemies_list, self.critic_feat_dim, 
                self.args.latent_dim, self.hidden_size, 
                args.transformer_heads, args.transformer_depth, 
                self.hidden_size, args.transformer_orth, device
            )
        elif self.args.algorithm_name == "subtask" or self.args.algorithm_name == "subtask_transfer":
            print("Critic: Construct subtask using cognition in a permutation invariance way")
            self.transformer = SubtaskTransformer(
                args, self.critic_feat_dim, self.args.latent_dim, self.hidden_size, 
                args.transformer_heads, args.transformer_depth, 
                self.hidden_size, args.transformer_orth, device
            )
        elif self.args.algorithm_name == "asn_gatten" or self.args.algorithm_name == "asn_gatten_transfer":
            print("Critic: ASN_Generalization_Attention")
            self.transformer = ASN_G_Atten(
                args, n_agents_list, n_enemies_list, self.critic_feat_dim, 
                self.args.latent_dim, self.hidden_size, 
                args.transformer_heads, args.transformer_depth, 
                self.hidden_size, args.transformer_orth, device
            )
        else:
            raise NotImplementedError

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
            raise NotImplementedError
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))
        self.to(device)

    def forward(self, cent_obs, rnn_states, masks, n_agents=None, n_enemies=None, is_loss=False):
        cent_obs = check(cent_obs, self.tpdv)
        rnn_states = check(rnn_states, self.tpdv)
        masks = check(masks, self.tpdv)
        
        if "subtask" in self.args.algorithm_name:
            # for trajectory record then analysis
            critic_feature_list, rnn_states, vae_loss_kl, vae_loss_re, entity_ob_list, subtask_list = self.transformer(
                cent_obs, rnn_states, masks, 
                n_agents, n_enemies,
                is_loss, collect_record=True, deterministic=False
            )
        else:
            critic_feature_list, rnn_states, vae_loss_kl, vae_loss_re = self.transformer(cent_obs, rnn_states, masks, n_agents, n_enemies, is_loss)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            raise NotImplementedError
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        
        values = self.v_out(torch.cat([torch.mean(critic_feature, dim=1) for critic_feature in critic_feature_list]))

        # values: (bs_na1 + bs_na2 + ..., 1)
        # rnn_states: a list of tensor
        if "subtask" in self.args.algorithm_name:
            return values, rnn_states, vae_loss_kl, vae_loss_re, subtask_list
            
        return values, rnn_states, vae_loss_kl, vae_loss_re


class Context_Encoder(nn.Module):
    def __init__(self, args, input_dim, latent_dim, device) -> None:
        super(Context_Encoder, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][args.use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), args.gain)

        self.args = args
        if args.use_obs_instead_of_state:
            self.critic_feat_dim = 16
        else:
            self.critic_feat_dim = 20
        self.actor_feat_dim = 16
        
        self.latent_dim = latent_dim
        self.hidden_size = args.hidden_size
        self.encoder = nn.Sequential(
            init_(nn.Linear(input_dim, self.hidden_size)),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            init_(nn.Linear(self.hidden_size, self.hidden_size)),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            init_(nn.Linear(self.hidden_size, 2*latent_dim))
        )

        self.decoder = nn.Sequential(
            init_(nn.Linear(latent_dim, self.hidden_size)),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            init_(nn.Linear(self.hidden_size, self.hidden_size)),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            init_(nn.Linear(self.hidden_size, input_dim))
        )

        self.tpdv = dict(dtype=torch.float32, device=device)

    def rp_sample(self, mus, sigmas):
        '''
        Reparameter sampling
        z ~ N(mu, sigma) --> (z-mu)/sqrt(sigma) ~ N(0, 1)
        '''
        epsilon = torch.distributions.Normal(
            check(torch.zeros(self.latent_dim), self.tpdv), check(torch.ones(self.latent_dim), self.tpdv)
        ).sample(mus.shape[:-1])
        z = mus + epsilon * torch.sqrt(sigmas)
        return z

    def encode(self, x, is_loss=False):
        '''
            x: (..., input_dim)
        '''
        params = self.encoder(x)
        mus, log_sigmas = torch.split(params, [self.latent_dim, self.latent_dim], dim=-1)
        sigmas = torch.exp(log_sigmas)
        '''
            z = torch.distributions.Normal(
                mus, sigmas
            ).rsample()
        '''
        z = self.rp_sample(mus, sigmas)
        x_z = torch.cat([x, z], dim=-1)

        if is_loss:
            loss_kl, loss_re, x_re = self.vae_loss(x, z, mus, sigmas)
        else:
            loss_kl, loss_re = None, None
            x_re = self.decode(z)
        return x_z, loss_kl, loss_re

    def decode(self, z):
        '''
        z: (..., output_dim)
        '''
        x = self.decoder(z)
        return x

    def kl_loss(self, mus, sigmas):
        '''
        mus_: (..., self.output_dim)
        sigmas_: (..., self.output_dim)
        '''
        '''
        # This method for computing kl_divergence is too slow as 
        # it compute kl_divergence between every N(mu, sigma) and N(0, 1) serially
        prior = torch.distributions.Normal(
            check(torch.zeros(self.output_dim), self.tpdv), check(torch.ones(self.output_dim), self.tpdv)
        )
        posteriors = [torch.distributions.Normal(mu, sigma) for mu, sigma in zip(torch.unbind(mus), torch.unbind(sigmas))]
        losses = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        loss_kl = torch.mean(torch.sum(torch.stack(losses), dim=-1), dim=0)
        '''
        '''
        # This loss is different from torch.distributions.kl.kl_divergence, why?
        # loss_kl = 0.5 * torch.mean(torch.sum(mus ** 2 + sigmas - torch.log(sigmas) - 1, dim=-1), dim=0)
        '''
        priors = torch.distributions.Normal(
            check(torch.zeros_like(mus), self.tpdv), check(torch.ones_like(sigmas), self.tpdv)
        )
        posteriors = torch.distributions.Normal(mus, sigmas)
        loss_kl = torch.sum(torch.distributions.kl.kl_divergence(posteriors, priors), dim=-1)
        return loss_kl

    def re_loss(self, x, z):
        '''
        x: (..., input_dim)
        z: (..., output_dim)
        '''
        x_re = self.decode(z)
        loss_re = torch.sum((x_re - x) ** 2, dim=-1)
        
        return loss_re, x_re

    def vae_loss(self, x, z, mus_, sigmas_):
        loss_kl = self.kl_loss(mus_, sigmas_)
        loss_re, x_re = self.re_loss(x, z)
        x_re = self.decode(z)
        return loss_kl, loss_re, x_re
        # return 0, loss_re, x_re

"""
Without Strategy: UPDeT
"""
class Eransformer(Transformer):
    def __init__(self, args, input_dim, latent_dim, emb_dim, heads, depth, output_dim, use_orthogonal, device) -> None:
        super(Transformer, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), args.gain)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.heads = heads
        self.input_embedding = init_(nn.Linear(input_dim, emb_dim*heads))

        tblocks = []
        for _ in range(depth):
            tblocks.append(
                TransformerBlock(args, emb_dim, heads, use_orthogonal)
            )
        self.tblocks = nn.Sequential(*tblocks)
        self.toprobs = init_(nn.Linear(emb_dim*heads, output_dim))
        self.tpdv = dict(dtype=torch.float32, device=device)
    
    def _encode(self, obs, n_agents, n_enemies):
        '''
        obs: a list of ob, in which ob's shape is (bs_na, obs_dim)
        '''
        feat_dim = self.input_dim
        bs_na_list = [ob.size(0) for ob in obs]
        
        split_lists = [[(n_agent+n_enemy)*feat_dim, 4] for n_agent, n_enemy in zip(n_agents, n_enemies)]
        entity_ob_list = [
            torch.split(ob, split_list, dim=-1)[0].contiguous().view(bs_na, n_agent+n_enemy, feat_dim).view(-1, feat_dim) \
                for ob, split_list, bs_na, n_agent, n_enemy in \
                    zip(obs, split_lists, bs_na_list, n_agents, n_enemies)
        ]
        entity_obs = torch.cat(entity_ob_list)
        return entity_obs, bs_na_list

    def _decode(self, entity_obs, n_agents, n_enemies, bs_na_list, feat_dim):
        '''
        obs is a tensor which shape is (-1, feat_dim)
        '''
        split_list = [bs_na * (n_agent + n_enemy) for bs_na, n_agent, n_enemy in zip(bs_na_list, n_agents, n_enemies)]
        entity_ob_list = torch.split(entity_obs, split_list, dim=0)
        entity_ob_list = [
            entity_ob.contiguous().view(bs_na, n_agent+n_enemy, feat_dim) \
                for entity_ob, bs_na, n_agent, n_enemy in zip(entity_ob_list, bs_na_list, n_agents, n_enemies)
        ]
        return entity_ob_list


    def forward(self, x, h, masks, n_agents, n_enemies, is_loss):
        entity_obs, bs_na_list = self._encode(x, n_agents, n_enemies)
        x = self.input_embedding(entity_obs)
        entity_ob_list = self._decode(x, n_agents, n_enemies, bs_na_list, self.emb_dim*self.heads)
        tb = self.tblocks(entity_ob_list)
        x = self.toprobs(tb)
        x_list = self._decode(x, n_agents, n_enemies, bs_na_list, self.output_dim)

        # vae loss: zero
        vae_loss_kl_list = [torch.tensor(0.).to(**self.tpdv) for _ in n_agents] if is_loss else None
        vae_loss_re_list = [torch.tensor(0.).to(**self.tpdv) for _ in n_agents] if is_loss else None
        
        return x_list, h, vae_loss_kl_list, vae_loss_re_list


"""
ASN: ACTION SEMANTICS NETWORK
Now Do Not Support Varying Obs/State/Action Space
"""
class ASN(nn.Module):
    def __init__(self, args, n_agents_list, n_enemies_list, input_dim, latent_dim, emb_dim, heads, depth, output_dim, use_orthogonal, device) -> None:
        super(ASN, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), args.gain)

        self.args = args
        assert len(n_agents_list) == 1 and len(n_enemies_list) == 1, \
            f"Now ASN Do Not Support Varying Obs/State/Action Space"
        self.feat_dim = input_dim
        self.n_agent = n_agents_list[0]
        self.n_enemy = n_enemies_list[0]
        self.n_entity = self.n_agent + self.n_enemy
        self.input_dim = input_dim * self.n_entity + 4
        # self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.heads = heads
        self.in_embedding = init_(nn.Linear(self.input_dim, self.emb_dim))
        self.out_embedding = init_(nn.Linear(self.feat_dim, self.emb_dim))
        # self.input_embedding = init_(nn.Linear(input_dim, emb_dim*heads))

        # tblocks = []
        # for _ in range(depth):
        #     tblocks.append(
        #         TransformerBlock(args, emb_dim, heads, use_orthogonal)
        #     )
        # self.tblocks = nn.Sequential(*tblocks)
        # self.toprobs = init_(nn.Linear(emb_dim*heads, output_dim))
        self.tpdv = dict(dtype=torch.float32, device=device)
    
    def _encode(self, obs, n_agents, n_enemies):
        '''
        obs: a list of ob, in which ob's shape is (bs_na, obs_dim)
        '''
        feat_dim = self.feat_dim
        bs_na_list = [ob.size(0) for ob in obs]
        
        split_lists = [[(n_agent+n_enemy)*feat_dim, 4] for n_agent, n_enemy in zip(n_agents, n_enemies)]
        entity_ob_list = [
            torch.split(ob, split_list, dim=-1)[0].contiguous().view(bs_na, n_agent+n_enemy, feat_dim) \
                for ob, split_list, bs_na, n_agent, n_enemy in \
                    zip(obs, split_lists, bs_na_list, n_agents, n_enemies)
        ]
        entity_obs = torch.cat(entity_ob_list)
        return entity_obs, bs_na_list

    def _decode(self, entity_obs, n_agents, n_enemies, bs_na_list, feat_dim):
        '''
        obs is a tensor which shape is (-1, feat_dim)
        '''
        split_list = [bs_na * (n_agent + n_enemy) for bs_na, n_agent, n_enemy in zip(bs_na_list, n_agents, n_enemies)]
        entity_ob_list = torch.split(entity_obs, split_list, dim=0)
        entity_ob_list = [
            entity_ob.contiguous().view(bs_na, n_agent+n_enemy, feat_dim) \
                for entity_ob, bs_na, n_agent, n_enemy in zip(entity_ob_list, bs_na_list, n_agents, n_enemies)
        ]
        return entity_ob_list


    def forward(self, x, h, masks, n_agents, n_enemies, is_loss):
        e_in = self.in_embedding(x[0]) # (bs_na, emb_dim)
        entity_obs, bs_na_list = self._encode(x, n_agents, n_enemies) # (bs_na, n_agent + n_enemy, emb_dim)

        n_enemy = n_enemies[0]
        # x = self.input_embedding(entity_obs)
        e_out = self.out_embedding(entity_obs[:, -n_enemy:, :]) # (bs_na, n_enemy, emb_dim)
        e_out = e_in.repeat(1, n_enemy).view(e_in.size(0), n_enemy, e_in.size(1)).view(*e_out.shape) * e_out # (bs_na, n_enemy, emb_dim)
        
        e = torch.cat([e_in.unsqueeze(1), e_out], dim=1) # (bs_na, 1+n_enemy, emb_dim)

        # entity_ob_list = self._decode(x, n_agents, n_enemies, bs_na_list, self.emb_dim*self.heads)
        # tb = self.tblocks(entity_ob_list)
        # x = self.toprobs(tb)
        # x_list = self._decode(x, n_agents, n_enemies, bs_na_list, self.output_dim)
        x_list = [e]

        # vae loss: zero
        vae_loss_kl_list = [torch.tensor(0.).to(**self.tpdv) for _ in n_agents] if is_loss else None
        vae_loss_re_list = [torch.tensor(0.).to(**self.tpdv) for _ in n_agents] if is_loss else None
        
        return x_list, h, vae_loss_kl_list, vae_loss_re_list


"""
ASNAttention for ASN_G
"""
class ASNAttention(nn.Module):
    def __init__(self, args, emb_dim=32, heads=4, use_orthogonal=True) -> None:
        # multi self attention(when heads>1)
        print("----------------------------------------------------------------------------------")
        print("=======================Entity  VAE Use Common SelfAttention=======================")
        print("----------------------------------------------------------------------------------")
        super(ASNAttention, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), args.gain)
        # q, k, v is same at the dim
        self.emb_dim = emb_dim
        self.heads = heads
        self.sqrt_emb_dim = self.emb_dim ** 0.5
        self.q_w = init_(nn.Linear(self.emb_dim * self.heads, self.emb_dim * self.heads))
        self.k_w = init_(nn.Linear(self.emb_dim * self.heads, self.emb_dim * self.heads))
        self.v_w = init_(nn.Linear(self.emb_dim * self.heads, self.emb_dim * self.heads))
        # self.head2emb = init_(nn.Linear(self.emb_dim * self.heads, self.emb_dim))
        # self.talking_w = init_(nn.Linear(args.n_agents+args.n_enemies, args.n_agents+args.n_enemies))
    
    def forward_i(self, features):
        # bs: batch_size, ne: num_eneities, fd: feature_dim, hs: heads
        bs, ne, _ = features.size()
        hs, fd = self.heads, self.emb_dim
        q_wave = self.q_w(features[:, :1, :]).view(bs, 1, hs, fd)
        k_wave = self.k_w(features).view(bs, ne, hs, fd)
        v_wave = self.v_w(features).view(bs, ne, hs, fd)

        q_wave = q_wave.transpose(1, 2).contiguous().view(bs*hs, 1, fd)
        k_wave = k_wave.transpose(1, 2).contiguous().view(bs*hs, ne, fd)
        v_wave = v_wave.transpose(1, 2).contiguous().view(bs*hs, ne, fd)

        # dot = F.softmax(self.talking_w(torch.matmul(q_wave, k_wave.transpose(-1, -2))/self.sqrt_emb_dim), dim=-1)
        dot = F.softmax(torch.matmul(q_wave, k_wave.transpose(-1, -2))/self.sqrt_emb_dim, dim=-1)
        features_attention = torch.matmul(dot, v_wave).view(bs, hs, 1, fd)
        features_attention = features_attention.transpose(1, 2).contiguous().view(bs, 1, hs*fd).squeeze(1)
        # features_attention = self.head2emb(features_attention) 
        return features_attention

    def forward(self, features):
        features_attention = [self.forward_i(feature) for feature in features]
        return features_attention
    
class ASN_G_Atten(nn.Module):
    def __init__(self, args, n_agents_list, n_enemies_list, input_dim, latent_dim, emb_dim, heads, depth, output_dim, use_orthogonal, device) -> None:
        super(ASN_G_Atten, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), args.gain)

        self.args = args
        # assert len(n_agents_list) == 1 and len(n_enemies_list) == 1, \
        #     f"Now ASN Do Not Support Varying Obs/State/Action Space"
        self.feat_dim = input_dim
        self.n_agent = n_agents_list
        self.n_enemy = n_enemies_list
        # self.n_entity = self.n_agent + self.n_enemy
        self.input_dim = input_dim
        # self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.heads = heads
        self.in_embedding = init_(nn.Linear(self.input_dim, self.emb_dim * self.heads))
        self.attention = ASNAttention(args, emb_dim, heads, True)
        self.out_embedding = init_(nn.Linear(self.feat_dim, self.emb_dim * self.heads))
        self.to_probs = init_(nn.Linear(self.emb_dim * self.heads, self.output_dim))
        self.tpdv = dict(dtype=torch.float32, device=device)
    
    def _encode(self, obs, n_agents, n_enemies):
        '''
        obs: a list of ob, in which ob's shape is (bs_na, obs_dim)
        '''
        feat_dim = self.feat_dim
        bs_na_list = [ob.size(0) for ob in obs]
        
        split_lists = [[(n_agent+n_enemy)*feat_dim, 4] for n_agent, n_enemy in zip(n_agents, n_enemies)]
        entity_ob_list = [
            torch.split(ob, split_list, dim=-1)[0].contiguous().view(bs_na, n_agent+n_enemy, feat_dim) \
                for ob, split_list, bs_na, n_agent, n_enemy in \
                    zip(obs, split_lists, bs_na_list, n_agents, n_enemies)
        ]
        # entity_obs = torch.cat(entity_ob_list)
        return entity_ob_list, bs_na_list

    def _decode(self, entity_obs, n_agents, n_enemies, bs_na_list, feat_dim):
        '''
        obs is a tensor which shape is (-1, feat_dim)
        '''
        split_list = [bs_na * (n_agent + n_enemy) for bs_na, n_agent, n_enemy in zip(bs_na_list, n_agents, n_enemies)]
        entity_ob_list = torch.split(entity_obs, split_list, dim=0)
        entity_ob_list = [
            entity_ob.contiguous().view(bs_na, n_agent+n_enemy, feat_dim) \
                for entity_ob, bs_na, n_agent, n_enemy in zip(entity_ob_list, bs_na_list, n_agents, n_enemies)
        ]
        return entity_ob_list


    def forward(self, x, h, masks, n_agents, n_enemies, is_loss):

        entity_ob_list, bs_na_list = self._encode(x, n_agents, n_enemies) # (bs_na, n_agent + n_enemy, emb_dim)

        e_in = [self.in_embedding(i_e) for i_e in entity_ob_list]
        e_in = self.attention(e_in) # (bs_na, emb_dim)
        e_out = [self.out_embedding(i_e_o[:, -n_enemy:, :]) for i_e_o, n_enemy in zip(entity_ob_list, self.n_enemy)] # (bs_na, n_enemy, emb_dim)
        e_out = [
            i_e_in.unsqueeze(1).repeat(1, n_enemy, 1) * i_e_out for i_e_in , i_e_out, n_enemy in zip(e_in, e_out, self.n_enemy)
        ]
        
        x_list = [torch.cat([i_e_in.unsqueeze(1), i_e_out], dim=1) for i_e_in, i_e_out in zip(e_in, e_out)] # (bs_na, 1+n_enemy, hs*emb_dim)
        x_list = [self.to_probs(i_x) for i_x in x_list]

        # vae loss: zero
        vae_loss_kl_list = [torch.tensor(0.).to(**self.tpdv) for _ in n_agents] if is_loss else None
        vae_loss_re_list = [torch.tensor(0.).to(**self.tpdv) for _ in n_agents] if is_loss else None
        
        return x_list, h, vae_loss_kl_list, vae_loss_re_list

"""
class for adaptive subtask semantics
"""
class Mod_Sequential(nn.Sequential):
    def forward4AS(self, x_emb, subtask_emb):
        for module in self:
            x_emb, subtask_emb, subtask_dot = module.forward_4_adaptive_semantics(x_emb, subtask_emb)
        return x_emb, subtask_emb, subtask_dot

"""
Permutation invariance Subtask
"""
class SubtaskTransformer(Transformer):
    def __init__(self, args, input_dim, latent_dim, emb_dim, heads, depth, output_dim, use_orthogonal, device) -> None:
        super(Transformer, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), args.gain)

        self.args = args
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.subtask_dim = args.num_subtask
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.heads = heads
        pi_choice = self.args.pi_choice # the choice of context_encoder

        tblocks = []
        for _ in range(depth):
            tblocks.append(
                TransformerBlock(args, emb_dim, heads, use_orthogonal)
            )
        # self.tblocks = nn.Sequential(*tblocks)
        self.tblocks = Mod_Sequential(*tblocks)
        if pi_choice == "mean_subtask":
            self.context_encoder = SubtaskContext_Encoder(args, input_dim, latent_dim, device)
            self.forward_func = self.forwardfor12
            self.input_embedding = init_(nn.Linear(input_dim + self.subtask_dim, emb_dim*heads))
            self.toprobs = init_(nn.Linear(emb_dim*heads, output_dim))
        elif pi_choice == "product_subtask":
            self.context_encoder = SubtaskContext_Encoder(args, input_dim, latent_dim, device)
            self.forward_func = self.forwardfor12
            self.input_embedding = init_(nn.Linear(input_dim + self.subtask_dim, emb_dim*heads))
            self.toprobs = init_(nn.Linear(emb_dim*heads, output_dim))
        elif pi_choice == "entity_cognition_cat":
            self.context_encoder = SubtaskConcatEntC_Context_Encoder(args, input_dim, latent_dim, device)
            self.forward_func = self.forwardfor3
            self.input_embedding = init_(nn.Linear(input_dim + self.latent_dim, emb_dim*heads))
            self.subtask_embedding = nn.Sequential(
                init_(nn.Linear(self.subtask_dim, self.latent_dim)), 
                init_(nn.Linear(self.latent_dim, self.latent_dim)), 
                nn.Tanh()
            )
            self.toprobs = init_(nn.Linear(emb_dim*heads + self.latent_dim, output_dim))
        elif pi_choice == "adaptive_semantics":
            print("Adaptive Subtask Semantics")
            self.context_encoder = SubtaskContext_Encoder(args, input_dim, latent_dim, device)
            self.forward_func = self.forward4AS
            self.input_embedding = init_(nn.Linear(input_dim, emb_dim*heads))
            self.subtask_embedding = nn.Sequential(
                init_(nn.Linear(self.subtask_dim, emb_dim)), 
                init_(nn.Linear(emb_dim, emb_dim * heads)), 
                nn.Tanh()
            )
            self.toprobs = init_(nn.Linear(2*emb_dim*heads, output_dim))
        else:
            raise NotImplementedError
        
        # self.toprobs = init_(nn.Linear(emb_dim*heads+self.subtask_dim, output_dim))
        self.tpdv = dict(dtype=torch.float32, device=device)
    
    def _encode(self, obs, n_agents, n_enemies):
        '''
        obs: a list of ob, in which ob's shape is (bs_na, obs_dim)
        '''
        feat_dim = self.input_dim
        bs_na_list = [ob.size(0) for ob in obs]
        
        if self.args.subtask_to_rnn:
            split_lists = [[(n_agent+n_enemy)*feat_dim, 4, self.subtask_dim] for n_agent, n_enemy in zip(n_agents, n_enemies)]
            past_subtask_list = [ob[:, -self.subtask_dim:] for ob in obs]
        else:
            split_lists = [[(n_agent+n_enemy)*feat_dim, 4] for n_agent, n_enemy in zip(n_agents, n_enemies)]
            past_subtask_list = [None for _ in n_agents]
        entity_ob_list = [
            torch.split(ob, split_list, dim=-1)[0].contiguous().view(bs_na, n_agent+n_enemy, feat_dim) \
                for ob, split_list, bs_na, n_agent, n_enemy in \
                    zip(obs, split_lists, bs_na_list, n_agents, n_enemies)
        ]
        # entity_obs = torch.cat(entity_ob_list)
        return entity_ob_list, past_subtask_list, bs_na_list

    def forward(self, x, h, masks, n_agents, n_enemies, is_loss, collect_record=False, deterministic=False):
        return self.forward_func(x, h, masks, n_agents, n_enemies, is_loss, collect_record, deterministic)

    def forwardfor12(self, x, h, masks, n_agents, n_enemies, is_loss, collect_record=False, deterministic=False):
        entity_ob_list, past_subtask_list, bs_na_list = self._encode(x, n_agents, n_enemies)
        vae_loss_kl, vae_loss_re, subtask_list, strategy_list, h_list = self.context_encoder.encode(entity_ob_list, past_subtask_list, h, masks, is_loss, n_agents, n_enemies, deterministic)
        
        entity_ob_subtask_list = [
            torch.cat([entity_ob, strategy], dim=-1) for entity_ob, strategy in zip(entity_ob_list, strategy_list)
        ]
        # x_list = [self.input_embedding(entity_ob) for entity_ob in entity_ob_list]
        x_list = [self.input_embedding(entity_ob_subtask) for entity_ob_subtask in entity_ob_subtask_list]
        
        tb = self.tblocks(x_list)
        # tb = torch.cat(
        #     [tb, torch.cat([strategy.view(-1, self.subtask_dim) for strategy in strategy_list], dim=0)],
        #     dim=-1
        # )
        tb_probs = self.toprobs(tb)
        x_list = self._decode(tb_probs, n_agents, n_enemies, bs_na_list, self.output_dim)

        if collect_record:
            return x_list, h_list, vae_loss_kl, vae_loss_re, entity_ob_list, subtask_list
        return x_list, h_list, vae_loss_kl, vae_loss_re

    def forwardfor3(self, x, h, masks, n_agents, n_enemies, is_loss, collect_record=False, deterministic=False):
        entity_ob_list, past_subtask_list, bs_na_list = self._encode(x, n_agents, n_enemies)
        entity_ob_cognition_list, vae_loss_kl, vae_loss_re, subtask_list, strategy_list, h_list = self.context_encoder.encode(entity_ob_list, past_subtask_list, h, masks, is_loss, n_agents, n_enemies, deterministic)
        
        x_list = [self.input_embedding(entity_ob_cognition) for entity_ob_cognition in entity_ob_cognition_list]
        
        tb = self.tblocks(x_list)
        subtask_emb_list = [self.subtask_embedding(strategy) for strategy in strategy_list]
        tb = torch.cat(
            [tb, torch.cat([subtask_emb.view(-1, self.latent_dim) for subtask_emb in subtask_emb_list], dim=0)],
            dim=-1
        )
        tb_probs = self.toprobs(tb)
        x_list = self._decode(tb_probs, n_agents, n_enemies, bs_na_list, self.output_dim)

        if collect_record:
            return x_list, h_list, vae_loss_kl, vae_loss_re, entity_ob_list, subtask_list
        return x_list, h_list, vae_loss_kl, vae_loss_re

    def forward4AS(self, x, h, masks, n_agents, n_enemies, is_loss, collect_record=False, deterministic=False):
        entity_ob_list, past_subtask_list, bs_na_list = self._encode(x, n_agents, n_enemies)
        vae_loss_kl, vae_loss_re, subtask_list, strategy_list, h_list = self.context_encoder.encode(entity_ob_list, past_subtask_list, h, masks, is_loss, n_agents, n_enemies, deterministic)
        
        subtask_emb = [self.subtask_embedding(subtask) for subtask in subtask_list]
        x_list = [self.input_embedding(entity_ob) for entity_ob in entity_ob_list]
        
        x_list, subtask_emb, subtask_dot = self.tblocks.forward4AS(x_list, subtask_emb)
        # tb = torch.cat(
        #     [tb, torch.cat([strategy.view(-1, self.subtask_dim) for strategy in strategy_list], dim=0)],
        #     dim=-1
        # )
        repeat_subtask_emb = [i_s.unsqueeze(1).repeat(1, n_agent+n_enemy, 1) for i_s, n_agent, n_enemy in zip(subtask_emb, n_agents, n_enemies)]
        tb_probs = [self.toprobs(torch.cat([i_x, i_s], dim=-1)) for i_x, i_s in zip(x_list, repeat_subtask_emb)]
        # tb_probs = torch.cat([i_t.view(-1, self.output_dim) for i_t in tb_probs])
        # x_list = self._decode(tb_probs, n_agents, n_enemies, bs_na_list, self.output_dim)
        x_list = tb_probs

        if self.args.record_attention:
            return x_list, h_list, vae_loss_kl, vae_loss_re, entity_ob_list, subtask_list, subtask_dot
        if collect_record:
            return x_list, h_list, vae_loss_kl, vae_loss_re, entity_ob_list, subtask_list
        return x_list, h_list, vae_loss_kl, vae_loss_re

class SubtaskContext_Encoder(Context_Encoder):
    def __init__(self, args, input_dim, latent_dim, device) -> None:
        super().__init__(args, input_dim, latent_dim, device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][args.use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), args.gain)
        self.num_subtask = self.args.num_subtask
        self.perm_invar_fc1 = init_(nn.Linear(latent_dim, latent_dim))
        
        if self.args.subtask_to_rnn:
            self.perm_invar_rnn = nn.GRU(latent_dim + self.num_subtask, latent_dim, num_layers=self.args.recurrent_N)
        else:
            self.perm_invar_rnn = nn.GRU(latent_dim, latent_dim, num_layers=self.args.recurrent_N)
        for name, param in self.perm_invar_rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if args.use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.perm_invar_fc2 = init_(nn.Linear(self.latent_dim, self.num_subtask))
        self.all_subtask_onehot = torch.eye(self.num_subtask).to(**self.tpdv)

    def perm_invar_encoder(self, x, p_s, hxs, masks):
        '''
            # train/intreaction
            x: (chunk_length*bs*na, (na+ne), feat_dim) / (threads*na, (na+ne), feat_dim)
            p_s: (chunk_length*bs*na, subtask_dim) / (threads*na, subtask_dim)
            hxs: (bs*na, recurrent_N, rnn_hidden_dim) / (threads*na, recurrent_N, rnn_hidden_dim)
            masks: (chunk_length*bs*na, 1) / (threads*na, 1)
        '''
        [x_bs, _, _] = x.size()
        [hxs_bs, rnn_layers, _] = hxs.size()
        assert masks.size(0) == x_bs

        # fc1
        x = self.perm_invar_fc1(x).mean(dim=-2) # (x_bs, latent_dim or feat_dim)
        if p_s is not None:
            x = torch.cat([x, p_s], dim=-1)

        if x_bs == hxs_bs:
            # print(f"Interactive: x's 1st shape is {x_bs} | h's 1st shape is {hxs_bs}")
            masks_re = masks.unsqueeze(-2).repeat(1, rnn_layers, 1) # (x_bs, rnn_layers, 1)
            x, hxs = self.perm_invar_rnn(x.unsqueeze(0),
                              (hxs * masks_re).transpose(0, 1).contiguous())
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, nae, feat_dim)
            # print(f"Training: x's 1st shape is {x_bs} | h's 1st shape is {hxs_bs}")
            N = hxs_bs
            T = int(x_bs / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]   # has_zeros = [0, 4, 7, ..., T]

            hxs = hxs.transpose(0, 1) # (recurrent_N, N, rnn_hidden_dim)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (hxs * masks[start_idx].view(1, -1, 1).repeat(rnn_layers, 1, 1)).contiguous() # (rnn_layers, N, rnn_hidden_dim)
                # (end-start, N, latent_dim) | (rnn_layers, N, rnn_hidden_dim)
                rnn_scores, hxs = self.perm_invar_rnn(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)

            # flatten
            x = x.reshape(T * N, -1) # (x_bs, latent_dim)
            hxs = hxs.transpose(0, 1) # (N, rnn_layers, rnn_hidden_dim) actually: rnn_hidden_dim = latent_dim

        # x = self.norm(x)
        return x, hxs
    
    def permutation_invariance_subtask(self, z_list, past_subtask, h_list, masks, n_agents, n_enemies, deterministic):
        # the shape of z in z_list is (bs_na, na+ne, fd)
        r_z_tuple, h_tuple = tuple(zip(*[self.perm_invar_encoder(z, p_s, h, mask) for z, p_s, h, mask in zip(z_list, past_subtask, h_list, masks)]))
        r_z_list = list(r_z_tuple) # (bs_na, na+ne, latent_dim) --> (bs_na, latent_dim)
        h_list = list(h_tuple) # (bs_na, latent_dim) --> (bs_na, latent_dim)

        # subtask_list = [torch.softmax(self.perm_invar_fc2(r_z), dim=-1) for r_z in r_z_list]
        subtask_list = [self.perm_invar_fc2(r_z) for r_z in r_z_list]

        if deterministic:
            subtask_list = [self.all_subtask_onehot[subtask.argmax(dim=-1)] for subtask in subtask_list]
        else:
            subtask_list = [self.all_subtask_onehot[F.gumbel_softmax(subtask, hard=True).argmax(dim=-1)] for subtask in subtask_list]
        repeat_subtask_list = [
            subtask.unsqueeze(-2).repeat(1, n_agent + n_enemy, 1) \
                for subtask, n_agent, n_enemy in zip(subtask_list, n_agents, n_enemies)
        ]
        return subtask_list, repeat_subtask_list, h_list


    def encode(self, x, past_subtask, h_list, masks, is_loss=False, n_agents=None, n_enemies=None, deterministic=False):
        '''
            x: [(bs_na, ne, input_dim), ...]
        '''
        params = [self.encoder(i_x) for i_x in x]
        z_list = []
        mu_list = []
        sigma_list = []
        for mus, log_sigmas in [torch.split(param, [self.latent_dim, self.latent_dim], dim=-1) for param in params]:
            sigmas = torch.exp(log_sigmas)
            z = self.rp_sample(mus, sigmas)
            z_list.append(z)
            mu_list.append(mus)
            sigma_list.append(sigmas)

        subtask_list, repeat_subtask_list, h_list = self.permutation_invariance_subtask(z_list, past_subtask, h_list, masks, n_agents, n_enemies, deterministic)
        # x_z = [torch.cat([i_x, i_subtask], dim=-1) for i_x, i_subtask in zip(x, subtask_list)]
        
        # x_z = [i_x_z.view(-1, *i_x_z.shape[2:]) for i_x_z in x_z] # (bs, na, na+ne, fd) --> (bs_na, na+ne, fd)
        # subtask_list = [i_subtask.view(-1, *i_subtask.shape[2:]) for i_subtask in subtask_list]

        if is_loss:
            loss_kl, loss_re, _ = self.vae_loss(x, z_list, mu_list, sigma_list, subtask_list, n_agents)
        else:
            loss_kl, loss_re = None, None
            # x_re = self.decode(z_list)
        # return x_z, loss_kl, loss_re, subtask_list
        return loss_kl, loss_re, subtask_list, repeat_subtask_list, h_list

    def re_loss(self, x, z):
        '''
        x: (..., input_dim)
        z: (..., output_dim)
        '''
        x_re = self.decode(z)
        loss_re = [torch.sum((ix_re - ix) ** 2, dim=-1).mean() for ix_re, ix in zip(x_re, x)]
        
        return loss_re, x_re

    def vae_loss(self, x, z, mus_, sigmas_, subtasks_, n_agents):
        loss_re, x_re = self.re_loss(x, z)
        if self.args.subtask_kl_loss:
            loss_kl = [0.001 * self.kl_loss(mu, sigma, subtask, n_agent) for mu, sigma, subtask, n_agent in zip(mus_, sigmas_, subtasks_, n_agents)]
        else:
            loss_kl = [torch.tensor(0.).to(**self.tpdv) for _ in loss_re]
        return loss_kl, loss_re, x_re

    def decode(self, z):
        '''
        z: (..., output_dim)
        '''
        x = [self.decoder(iz) for iz in z]
        return x
    
    def kl_loss(self, mus, sigmas, subtask, n_agent):
        '''
            mus: (bs_na, ne, latent_dim)
            sigmas_squared: (bs_na, ne, latent_dim)
            subtask: (bs_na, subtask_dim)
        '''
        mu, sigma = self._product_of_cognition_on_entity(mus, sigmas)
        loss_kl = self._subtask_consistent_loss(mu, sigma, subtask, n_agent)
        return loss_kl


    def _product_of_cognition_on_entity(self, mus, sigmas_squared):
        '''
            compute the product of cognition_on_entity, so the result is cognition_on_env
            mus: (bs_na, ne, latent_dim)
            sigmas_squared: (bs_na, ne, latent_dim)
        '''
        mu, sigma_squared = self._product_of_gaussian(mus, sigmas_squared, masks=None)
        return mu, sigma_squared

    def _subtask_consistent_loss(self, mus, sigmas_squared, subtask, n_agent):
        '''
            compute the product of cognition_on_env of agents with same subtask, where 
            the product is named product_env, and compute the kl_divergence of cognition_on_env 
            with its corresponding product_env as a subtask consistent loss
            mus: (bs_na, latent_dim)
            sigmas_squared: (bs_na, latent_dim)
            subtask: (bs_na, subtask_dim)
        '''
        # (bs_na, latent_dim) --> (bs, na, latent_dim)

        mus = mus.view(int(mus.size(0) / n_agent), n_agent, mus.size(-1))
        sigmas_squared = sigmas_squared.view(int(sigmas_squared.size(0) / n_agent), n_agent, sigmas_squared.size(-1))
        subtask = subtask.view(int(subtask.size(0) / n_agent), n_agent, subtask.size(-1))
        
        sum_subtask = torch.sum(subtask, dim=-2) # (bs, subtask_dim)
        
        loss = check(torch.tensor(0.), self.tpdv)
        for idx_subtask in range(self.num_subtask):
            idx_bs = sum_subtask[:, idx_subtask] > 1
            masks = subtask[idx_bs, :, idx_subtask]
            mus_ = mus[idx_bs, :, :]
            sigmas_squared_ = sigmas_squared[idx_bs, :, :]
            mu, sigma_squared = self._product_of_gaussian(mus_, sigmas_squared_, masks)
            # loss compute
            mu = mu.unsqueeze(-2).repeat(1, n_agent, 1)
            sigma_squared = sigma_squared.unsqueeze(-2).repeat(1, n_agent, 1)
            posterior = torch.distributions.Normal(mus_, sigmas_squared_)
            prior = torch.distributions.Normal(mu, sigma_squared)
            loss = loss + (torch.distributions.kl.kl_divergence(posterior, prior).mean(-1) * masks).sum() / masks.sum()
        
        loss = loss / self.num_subtask

        return loss

    def _product_of_gaussian(self, mus, sigmas_squared, masks):
        '''
            compute mu, sigma of product of gaussians
            mus: (bs_na, ne, latent_dim) or (bs, na, latent_dim)
            sigmas_squared: (bs_na, ne, latent_dim) or (bs, na, latent_dim)
            masks: (ba_na, ne, latent_dim) or (bs, na, latent_dim)
        '''
        if masks is None:
            sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
            sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=-2)
            mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=-2)
        else:
            masks = masks.unsqueeze(-1).repeat(1, 1, mus.size(-1))
            sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
            sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared) * masks, dim=-2)
            mu = sigma_squared * torch.sum((mus / sigmas_squared) * masks, dim=-2)
        # mu: (bs_na, latent_dim) or (bs, latent_dim) | sigma_squared: (bs_na, latent_dim) or (bs, latent_dim)
        return mu, sigma_squared


class SubtaskContext_Encoder(SubtaskContext_Encoder):
    def __init__(self, args, input_dim, latent_dim, device) -> None:
        super().__init__(args, input_dim, latent_dim, device)
        del self.perm_invar_fc1
        del self.perm_invar_fc2
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][args.use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), args.gain)
        self.perm_invar_fc = init_(nn.Linear(self.latent_dim, self.num_subtask))

    def perm_invar_encoder(self, x, p_s, hxs, masks):
        '''
            # train/intreaction
            x: (chunk_length*bs*na, feat_dim) / (threads*na, feat_dim)
            p_s: (chunk_length*bs*na, subtask_dim) / (threads*na, subtask_dim)
            hxs: (bs*na, recurrent_N, rnn_hidden_dim) / (threads*na, recurrent_N, rnn_hidden_dim)
            masks: (chunk_length*bs*na, 1) / (threads*na, 1)
        '''
        [x_bs, _] = x.size()
        [hxs_bs, rnn_layers, _] = hxs.size()
        assert masks.size(0) == x_bs

        # fc1
        # x = self.perm_invar_fc1(x).mean(dim=-2) # (x_bs, latent_dim or feat_dim)
        if p_s is not None:
            x = torch.cat([x, p_s], dim=-1)

        if x_bs == hxs_bs:
            # print(f"Interactive: x's 1st shape is {x_bs} | h's 1st shape is {hxs_bs}")
            masks_re = masks.unsqueeze(-2).repeat(1, rnn_layers, 1) # (x_bs, rnn_layers, 1)
            x, hxs = self.perm_invar_rnn(x.unsqueeze(0),
                              (hxs * masks_re).transpose(0, 1).contiguous())
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, nae, feat_dim)
            # print(f"Training: x's 1st shape is {x_bs} | h's 1st shape is {hxs_bs}")
            N = hxs_bs
            T = int(x_bs / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]   # has_zeros = [0, 4, 7, ..., T]

            hxs = hxs.transpose(0, 1) # (recurrent_N, N, rnn_hidden_dim)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (hxs * masks[start_idx].view(1, -1, 1).repeat(rnn_layers, 1, 1)).contiguous() # (rnn_layers, N, rnn_hidden_dim)
                # (end-start, N, latent_dim) | (rnn_layers, N, rnn_hidden_dim)
                rnn_scores, hxs = self.perm_invar_rnn(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)

            # flatten
            x = x.reshape(T * N, -1) # (x_bs, latent_dim)
            hxs = hxs.transpose(0, 1) # (N, rnn_layers, rnn_hidden_dim) actually: rnn_hidden_dim = latent_dim

        # x = self.norm(x)
        return x, hxs

    def encode(self, x, past_subtask, h_list, masks, is_loss=False, n_agents=None, n_enemies=None, deterministic=False):
        '''
            x: [(bs_na, ne, input_dim), ...]
        '''
        params = [self.encoder(i_x) for i_x in x]
        z4re_list = []
        z4ff_list = []
        # mu_list = []
        # sigma_list = []
        g_mu_list = []
        g_sigma_list = []
        for mus, log_sigmas in [torch.split(param, [self.latent_dim, self.latent_dim], dim=-1) for param in params]:
            sigmas = torch.exp(log_sigmas)
            z4re = self.rp_sample(mus, sigmas) # entity-cognition for reconstructing entity-obs
            z4re_list.append(z4re)
            g_mu, g_sigma = self.cognition_product_of_gaussians(mus, sigmas)
            z4ff = self.rp_sample(g_mu, g_sigma) # env-cognition for constructing subtask
            z4ff_list.append(z4ff)
            g_mu_list.append(g_mu)
            g_sigma_list.append(g_sigma)
            # mu_list.append(mus)
            # sigma_list.append(sigmas)

        subtask_list, repeat_subtask_list, h_list = self.permutation_invariance_subtask(z4ff_list, past_subtask, h_list, masks, n_agents, n_enemies, deterministic)
        # x_z = [torch.cat([i_x, i_subtask], dim=-1) for i_x, i_subtask in zip(x, subtask_list)]
        
        # x_z = [i_x_z.view(-1, *i_x_z.shape[2:]) for i_x_z in x_z] # (bs, na, na+ne, fd) --> (bs_na, na+ne, fd)
        # subtask_list = [i_subtask.view(-1, *i_subtask.shape[2:]) for i_subtask in subtask_list]

        if is_loss:
            loss_kl, loss_re, _ = self.vae_loss(x, z4re_list, g_mu_list, g_sigma_list, subtask_list, n_agents)
        else:
            loss_kl, loss_re = None, None
            # x_re = self.decode(z_list)
        # return x_z, loss_kl, loss_re, subtask_list
        return loss_kl, loss_re, subtask_list, repeat_subtask_list, h_list

    def cognition_product_of_gaussians(self, mus, sigmas_squared):
        '''
        compute mu, sigma of product of gaussians
        '''
        sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
        sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=-2)
        mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=-2)
        # repeat_mu = mu.unsqueeze(-2).repeat(1, mus.size(-2), 1)
        # repeat_sigma_squared = sigma_squared.unsqueeze(-2).repeat(1, sigmas_squared.size(-2), 1)
        # mu = torch.stack([mu for _ in range(mus.size(-2))], dim=-2)
        # sigma_squared = torch.stack([sigma_squared for _ in range(sigmas_squared.size(-2))], dim=-2)
        # return mu, sigma_squared, repeat_mu, repeat_sigma_squared
        return mu, sigma_squared
    
    def permutation_invariance_subtask(self, z_list, past_subtask, h_list, masks, n_agents, n_enemies, deterministic):
        # the shape of z in z_list is (bs_na, na+ne, fd)
        r_z_tuple, h_tuple = tuple(zip(*[self.perm_invar_encoder(z, p_s, h, mask) for z, p_s, h, mask in zip(z_list, past_subtask, h_list, masks)]))
        r_z_list = list(r_z_tuple) # (bs_na, na+ne, latent_dim) --> (bs_na, latent_dim)
        h_list = list(h_tuple) # (bs_na, latent_dim) --> (bs_na, latent_dim)

        # subtask_list = [torch.softmax(self.perm_invar_fc2(r_z), dim=-1) for r_z in r_z_list]
        subtask_list = [self.perm_invar_fc(r_z) for r_z in r_z_list]

        if deterministic:
            # print(f"permutation_invariance_subtask: deterministic is True")
            subtask_list = [self.all_subtask_onehot[subtask.argmax(dim=-1)] for subtask in subtask_list]
        else:
            # print(f"permutation_invariance_subtask: deterministic is False")
            subtask_list = [F.gumbel_softmax(subtask, hard=True) for subtask in subtask_list]
        repeat_subtask_list = [
            subtask.unsqueeze(-2).repeat(1, n_agent + n_enemy, 1) \
                for subtask, n_agent, n_enemy in zip(subtask_list, n_agents, n_enemies)
        ]
        return subtask_list, repeat_subtask_list, h_list

    def kl_loss(self, mus, sigmas, subtask, n_agent):
        '''
            mus: (bs_na, latent_dim)
            sigmas_squared: (bs_na, latent_dim)
            subtask: (bs_na, subtask_dim)
        '''
        # mu, sigma = self._product_of_cognition_on_entity(mus, sigmas)
        loss_kl = self._subtask_consistent_loss(mus, sigmas, subtask, n_agent)
        return loss_kl


class SubtaskConcatEntC_Context_Encoder(SubtaskContext_Encoder):
    def __init__(self, args, input_dim, latent_dim, device) -> None:
        super().__init__(args, input_dim, latent_dim, device)

    def encode(self, x, past_subtask, h_list, masks, is_loss=False, n_agents=None, n_enemies=None, deterministic=False):
        '''
            x: [(bs_na, ne, input_dim), ...]
        '''
        params = [self.encoder(i_x) for i_x in x]
        z4re_list = []
        z4ff_list = []
        g_mu_list = []
        g_sigma_list = []
        for mus, log_sigmas in [torch.split(param, [self.latent_dim, self.latent_dim], dim=-1) for param in params]:
            sigmas = torch.exp(log_sigmas)
            z4re = self.rp_sample(mus, sigmas) # entity-cognition for reconstructing entity-obs
            z4re_list.append(z4re)
            g_mu, g_sigma = self.cognition_product_of_gaussians(mus, sigmas)
            z4ff = self.rp_sample(g_mu, g_sigma) # env-cognition for constructing subtask
            z4ff_list.append(z4ff)
            g_mu_list.append(g_mu)
            g_sigma_list.append(g_sigma)

        subtask_list, repeat_subtask_list, h_list = self.permutation_invariance_subtask(z4ff_list, past_subtask, h_list, masks, n_agents, n_enemies, deterministic)
        # x_z = [torch.cat([i_x, i_subtask], dim=-1) for i_x, i_subtask in zip(x, subtask_list)]
        x_z = [torch.cat([i_x, i_z], dim=-1) for i_x, i_z in zip(x, z4re_list)]

        if is_loss:
            loss_kl, loss_re, _ = self.vae_loss(x, z4re_list, g_mu_list, g_sigma_list, subtask_list, n_agents)
        else:
            loss_kl, loss_re = None, None
            # x_re = self.decode(z_list)
        # return x_z, loss_kl, loss_re, subtask_list
        return x_z, loss_kl, loss_re, subtask_list, repeat_subtask_list, h_list