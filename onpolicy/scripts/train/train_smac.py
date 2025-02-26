#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.envs.starcraft2.smac_maps import get_map_params
from onpolicy.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from copy import deepcopy
from onpolicy.utils.multi_envs_shared_buffer import MultiEnvShareSubprocVecEnv

import random

"""Train script for SMAC."""

def make_train_env(all_args):
    if all_args.use_unified_env:
        from onpolicy.envs.starcraft2.StarCraft2_UnifiedEnv import StarCraft2UnifiedEnv as StarCraft2Env
        # from onpolicy.envs.starcraft2.sc_env.smac_wrapper import SMAC2 as StarCraft2Env
    else:
        from onpolicy.envs.starcraft2.StarCraft2_Env import StarCraft2Env
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    def get_multi_env_fn(rank, args_copy):
        def init_env():
            if args_copy.env_name == "StarCraft2":
                env = StarCraft2Env(args_copy)
            else:
                print("Can not support the " + args_copy.env_name + "environment.")
                raise NotImplementedError
            env.seed(args_copy.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        if all_args.multi_envs is None:
            return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
        else:
            # multi_envs is concatenate from multi map_name by |
            multi_envs = all_args.multi_envs.split('|')
            assert len(multi_envs) == len(set(multi_envs)), f"map_name in multi_envs cannot be same"
            assert all_args.n_rollout_threads % len(multi_envs) == 0, \
                f"n_rollout_threads {all_args.n_rollout_threads} should divided by num of multi_envs {len(multi_envs)}"
            num_threads_per_env = all_args.n_rollout_threads // len(multi_envs)
            print(f"num_threads_per_env: {num_threads_per_env}")
            multi_env_list = []
            n_agents_list = []
            n_enemies_list = []
            current_thread = 0
            for map_name in multi_envs:
                print(f"map_name: {map_name}")
                seed_thread = 0
                args_copy = deepcopy(all_args)
                args_copy.map_name = map_name
                multi_env_list.append([get_multi_env_fn(seed_thread + i, args_copy) for i in range(num_threads_per_env)])
                n_agents_list.append(get_map_params(map_name)["n_agents"])
                n_enemies_list.append(get_map_params(map_name)["n_enemies"])
                current_thread += num_threads_per_env
            assert current_thread == all_args.n_rollout_threads
            return MultiEnvShareSubprocVecEnv(multi_env_list, num_threads_per_env, multi_envs, n_agents_list, n_enemies_list)


def make_eval_env(all_args):
    if all_args.use_unified_env:
        from onpolicy.envs.starcraft2.StarCraft2_UnifiedEnv import StarCraft2UnifiedEnv as StarCraft2Env
        # from onpolicy.envs.starcraft2.sc_env.smac_wrapper import SMAC2 as StarCraft2Env
    else:
        from onpolicy.envs.starcraft2.StarCraft2_Env import StarCraft2Env
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    def get_multi_env_fn(rank, args_copy):
        def init_env():
            if args_copy.env_name == "StarCraft2":
                env = StarCraft2Env(args_copy)
            else:
                print("Can not support the " + args_copy.env_name + "environment.")
                raise NotImplementedError
            env.seed(args_copy.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.multi_envs is None:
        if all_args.n_eval_rollout_threads == 1:
            return ShareDummyVecEnv([get_env_fn(0)])
        else:
            return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])
    else:
        # multi_envs is concatenate from multi map_name by |
        multi_envs = all_args.multi_envs.split('|')
        # args_copy.n_eval_rollout_threads = len(multi_envs)
        multi_env_list = []
        n_agents_list = []
        n_enemies_list = []
        for map_name in multi_envs:
            args_copy = deepcopy(all_args)
            args_copy.map_name = map_name
            args_copy.map_name = map_name
            multi_env_list.append(get_multi_env_fn(0, args_copy))
            n_agents_list.append(get_map_params(map_name)["n_agents"])
            n_enemies_list.append(get_map_params(map_name)["n_enemies"])
        return MultiEnvShareSubprocVecEnv(multi_env_list, 1, multi_envs, n_agents_list, n_enemies_list)


def parse_args(args, parser):
    parser.add_argument('--map_name', type=str, default='3m',
                        help="Which smac map to run on")
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_false', default=True)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_false', default=True)

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    # ## debug
    # from onpolicy.debug_config import debug_get_config
    # parser = debug_get_config()
    # train
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        # assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo" \
        or all_args.algorithm_name == "entity" or all_args.algorithm_name == "entity_transfer" \
            or all_args.algorithm_name == "asn" \
                or all_args.algorithm_name == "subtask" or all_args.algorithm_name == "subtask_transfer" \
                        or all_args.algorithm_name == "asn_gatten" or all_args.algorithm_name == "asn_gatten_transfer":
        # assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
        #     "check recurrent policy!")
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    if all_args.multi_envs is None:
        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                        0] + "/results") / all_args.env_name / all_args.map_name / all_args.algorithm_name / all_args.experiment_name
    else:
        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                            0] + "/results") / all_args.env_name / all_args.multi_envs.replace("|", "-") / all_args.algorithm_name / all_args.experiment_name
    if all_args.experiment_name == "evaluation" or "transfer" in all_args.algorithm_name:
        run_dir = run_dir / all_args.model_dir.split("/")[-5] / all_args.model_dir.split("/")[-3]
    
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.map_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)

    # env
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = get_map_params(all_args.map_name)["n_agents"]
    num_enemies = get_map_params(all_args.map_name)["n_enemies"]
    all_args.n_agents = num_agents
    all_args.n_enemies = num_enemies

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        if all_args.multi_envs is None:
            from onpolicy.runner.shared.smac_runner import SMACRunner as Runner
        else:
            if all_args.algorithm_name == "vae":
                from onpolicy.runner.shared.vae_smac_runner import VAESMACRunner as Runner
            elif all_args.algorithm_name == "pearl":
                from onpolicy.runner.shared.pearl_multi_smac_runner import PearlMultiSMACRunner as Runner
            elif all_args.algorithm_name == "new_pearl" or all_args.algorithm_name == "pearl_transfer":
                from onpolicy.runner.shared.p_smac_runner import PSMACRunner as Runner
            elif "maml" in all_args.algorithm_name:
                from onpolicy.runner.shared.maml_smac_runner import MAMLSMACRunner as Runner
            # elif all_args.algorithm_name == "entity" or all_args.algorithm_name == "entity_transfer":
            #     from onpolicy.runner.shared.entity_smac_runner import EntitySMACRunner as Runner
            elif all_args.algorithm_name == "entity" \
                    or all_args.algorithm_name == "entity_transfer" \
                                or all_args.algorithm_name == "asn" \
                                    or all_args.algorithm_name == "subtask" \
                                        or all_args.algorithm_name == "subtask_transfer" \
                                                    or all_args.algorithm_name == "asn_gatten" \
                                                        or all_args.algorithm_name == "asn_gatten_transfer": \
                from onpolicy.runner.shared.ev_smac_runner import EntityVAESMACRunner as Runner
            else:
                # from onpolicy.runner.shared.multi_smac_runner import MultiSMACRunner as Runner
                from onpolicy.runner.shared.t_smac_runner import TSMACRunner as Runner
    else:
        from onpolicy.runner.separated.smac_runner import SMACRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
