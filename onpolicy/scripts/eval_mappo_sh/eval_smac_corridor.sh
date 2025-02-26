#!/bin/sh
env="StarCraft2"
map="corridor"
algo="mappo"
exp="evaluation"
seed_max=30
model_dir=../results/StarCraft2/corridor/mappo/check/run1/models

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
echo "seed is ${seed_max}:"
CUDA_VISIBLE_DEVICES=1 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--map_name ${map} --seed ${seed_max} --n_training_threads 1 --n_rollout_threads 2 --num_mini_batch 1 --episode_length 400 \
--num_env_steps 10000000 --ppo_epoch 5 --use_value_active_masks --use_eval --eval_episodes 32 --only_evaluation 1 --save_replay 1 \
--model_dir ${model_dir}