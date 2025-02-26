#!/bin/sh
env="StarCraft2"
map="27m_vs_30m"
algo="rmappo"
exp="check"
seed_max=4

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
echo "seed is ${seed_max}:"
CUDA_VISIBLE_DEVICES=3 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--map_name ${map} --seed ${seed_max} --n_training_threads 1 --n_rollout_threads 16 --num_mini_batch 1 --episode_length 400 \
--num_env_steps 10000000 --ppo_epoch 5 --use_value_active_masks --use_eval --eval_episodes 32

