#!/bin/sh
env="StarCraft2"
map="3m"
multi_envs="2s3z"
algo="asn"
exp="mlp"
seed_max=4
transformer=1
recurrent_policy=0
hidden_size=64
transformer_heads=3
transformer_depth=1
shared_transformer=0
atrous_attention=0
num_mini_batch=8

use_pearl=1
kl_lambda=1
use_reconstruct_loss=1
use_actor_loss=1
context_map_ohid=0
recent_context=0
latent_dim=8

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, 
max seed is ${seed_max}, transformer is ${transformer}, recurrent policy is ${recurrent_policy}, 
hidden size is ${hidden_size}, heads is ${transformer_heads}, transformer depth is ${transformer_depth}, 
shared transformer is ${shared_transformer}, atrous attention is ${atrous_attention}, multi envs is ${multi_envs}"

echo "seed is ${seed_max}:"
CUDA_VISIBLE_DEVICES=3 python ../train/train_smac.py --env_name ${env} \
--algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} \
--seed ${seed_max} --use_recurrent_policy ${recurrent_policy} \
--use_unified_env ${transformer} --entity_feature ${transformer} --transformer_actor ${transformer} \
--transformer_critic ${transformer} --transformer_orth ${transformer} \
--hidden_size ${hidden_size} --transformer_heads ${transformer_heads} \
--transformer_depth ${transformer_depth} --shared_transformer ${shared_transformer} \
--atrous_attention ${atrous_attention} --multi_envs ${multi_envs} \
--use_pearl ${use_pearl} --recent_context ${recent_context} --context_map_ohid ${context_map_ohid} \
--use_reconstruct_loss ${use_reconstruct_loss} --use_actor_loss ${use_actor_loss} --kl_lambda ${kl_lambda} \
--latent_dim ${latent_dim} \
--n_training_threads 1 --n_rollout_threads 16 --num_mini_batch ${num_mini_batch} --episode_length 400 --num_env_steps 10000000 --ppo_epoch 5 --use_value_active_masks --use_eval \
--use_linear_lr_decay
