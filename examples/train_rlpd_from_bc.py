#!/usr/bin/env python3
"""
BC -> RL Fine-tune 脚本

用法:
1. 先用 train_bc.py 训练 BC checkpoint
2. 用本脚本从 BC checkpoint 初始化 SAC agent 开始 RL 训练

python train_rlpd_from_bc.py \
    --exp_name usb_pickup_insertion \
    --checkpoint_path /path/to/new/rl_checkpoint \
    --bc_checkpoint_path /path/to/bc_checkpoint \
    --demo_path /path/to/demo.pkl \
    --learner
"""

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
from flax.core import freeze, unfreeze
import os
import copy
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_bc_agent,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_string("bc_checkpoint_path", None, "Path to BC pretrained checkpoint.")
flags.DEFINE_string("wandb_run_id", None, "Wandb run ID for resuming (optional).")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_boolean("debug", False, "Debug mode.")


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


def print_red(x):
    return print("\033[91m {}\033[00m".format(x))


##############################################################################
# BC -> SAC 权重迁移
##############################################################################


def transfer_bc_to_sac(bc_agent: BCAgent, sac_agent, bc_checkpoint_path: str):
    """
    将 BC checkpoint 的 actor 权重迁移到 SAC agent
    
    HIL-SERL 的实际结构：
    - BC params 顶层 key: 'modules_actor'
    - SAC params 顶层 keys: 'modules_actor', 'modules_critic', 'modules_target_critic', etc.
    
    迁移策略：直接用 BC 的 'modules_actor' 替换 SAC 的 'modules_actor'
    """
    # 1. 加载 BC checkpoint
    print_yellow(f"Loading BC checkpoint from: {bc_checkpoint_path}")
    bc_ckpt = checkpoints.restore_checkpoint(
        bc_checkpoint_path,
        target=bc_agent.state,
    )
    
    # 2. 打印结构信息用于调试
    bc_keys = list(bc_ckpt.params.keys())
    sac_keys = list(sac_agent.state.params.keys())
    print_yellow(f"BC checkpoint params top-level keys: {bc_keys}")
    print_yellow(f"SAC agent params top-level keys: {sac_keys}")
    
    # 3. 找到 actor 对应的 key
    # HIL-SERL 使用 'modules_actor' 作为 key
    actor_key = None
    for key in ['modules_actor', 'actor']:
        if key in bc_keys and key in sac_keys:
            actor_key = key
            break
    
    if actor_key is None:
        # 如果 BC 只有一个 key，假设它就是 actor
        if len(bc_keys) == 1:
            bc_actor_key = bc_keys[0]
            # 在 SAC 中找包含 'actor' 的 key
            sac_actor_key = None
            for key in sac_keys:
                if 'actor' in key.lower():
                    sac_actor_key = key
                    break
            if sac_actor_key:
                print_yellow(f"Mapping BC key '{bc_actor_key}' -> SAC key '{sac_actor_key}'")
                actor_key = sac_actor_key
                bc_actor_params = bc_ckpt.params[bc_actor_key]
            else:
                print_red("Cannot find actor key in SAC params!")
                return sac_agent
        else:
            print_red(f"Cannot find matching actor key! BC keys: {bc_keys}, SAC keys: {sac_keys}")
            return sac_agent
    else:
        bc_actor_params = bc_ckpt.params[actor_key]
    
    print_green(f"Found actor key: '{actor_key}'")
    
    # 4. 验证结构匹配
    bc_actor_leaves = jax.tree_util.tree_leaves(bc_actor_params)
    sac_actor_leaves = jax.tree_util.tree_leaves(sac_agent.state.params[actor_key])
    
    print_yellow(f"BC actor has {len(bc_actor_leaves)} parameter arrays")
    print_yellow(f"SAC actor has {len(sac_actor_leaves)} parameter arrays")
    
    if len(bc_actor_leaves) != len(sac_actor_leaves):
        print_red(f"Parameter count mismatch!")
        print_yellow("Attempting key-based transfer...")
        return transfer_by_keys(bc_actor_params, sac_agent, actor_key)
    
    # 检查 shapes
    shapes_match = True
    for i, (bc_leaf, sac_leaf) in enumerate(zip(bc_actor_leaves, sac_actor_leaves)):
        if bc_leaf.shape != sac_leaf.shape:
            print_red(f"Shape mismatch at index {i}: BC {bc_leaf.shape} vs SAC {sac_leaf.shape}")
            shapes_match = False
    
    if not shapes_match:
        print_red("Shape mismatch detected!")
        print_yellow("Attempting key-based transfer...")
        return transfer_by_keys(bc_actor_params, sac_agent, actor_key)
    
    # 5. 执行迁移
    print_green("✓ BC and SAC actor structures match perfectly!")
    
    # 需要处理 FrozenDict
    new_params = unfreeze(sac_agent.state.params)
    new_params[actor_key] = bc_actor_params
    new_params = freeze(new_params)
    
    new_state = sac_agent.state.replace(params=new_params)
    sac_agent = sac_agent.replace(state=new_state)
    
    print_green("✓ Successfully transferred BC weights to SAC actor!")
    return sac_agent


def transfer_by_keys(bc_actor_params, sac_agent, actor_key):
    """
    按 key 名称匹配迁移权重（更鲁棒的方法）
    保持原始 pytree 结构（FrozenDict vs dict）
    """
    from flax.core import FrozenDict
    
    transferred_count = [0]
    
    def recursive_transfer(bc_node, sac_node, path=""):
        """
        递归迁移匹配的参数。
        关键：必须保持与 sac_node 完全相同的结构类型！
        """
        # Handle FrozenDict - must return FrozenDict
        if isinstance(sac_node, FrozenDict):
            result = {}
            bc_dict = dict(bc_node) if isinstance(bc_node, (dict, FrozenDict)) else {}
            for key in sac_node:
                current_path = f"{path}/{key}" if path else key
                if key in bc_dict:
                    result[key] = recursive_transfer(bc_dict[key], sac_node[key], current_path)
                else:
                    result[key] = sac_node[key]
            return FrozenDict(result)  # 保持 FrozenDict 类型
        
        # Handle regular dict
        elif isinstance(sac_node, dict):
            result = {}
            bc_dict = dict(bc_node) if isinstance(bc_node, (dict, FrozenDict)) else {}
            for key in sac_node:
                current_path = f"{path}/{key}" if path else key
                if key in bc_dict:
                    result[key] = recursive_transfer(bc_dict[key], sac_node[key], current_path)
                else:
                    result[key] = sac_node[key]
            return result
        
        # Leaf node
        else:
            if hasattr(sac_node, 'shape') and hasattr(bc_node, 'shape'):
                if sac_node.shape == bc_node.shape:
                    transferred_count[0] += 1
                    print_green(f"  ✓ Transferred: {path}")
                    return bc_node
                else:
                    print_yellow(f"  ✗ Shape mismatch: {path} "
                               f"(BC: {bc_node.shape}, SAC: {sac_node.shape})")
            return sac_node
    
    print_yellow("Attempting key-based parameter transfer:")
    
    sac_params = sac_agent.state.params
    new_actor_params = recursive_transfer(bc_actor_params, sac_params[actor_key])
    
    # Rebuild params preserving top-level structure
    if isinstance(sac_params, FrozenDict):
        new_params = FrozenDict({
            **{k: v for k, v in sac_params.items() if k != actor_key},
            actor_key: new_actor_params
        })
    else:
        new_params = {**sac_params, actor_key: new_actor_params}
    
    print_green(f"✓ Transferred {transferred_count[0]} parameter arrays")
    
    new_state = sac_agent.state.replace(params=new_params)
    return sac_agent.replace(state=new_state)


##############################################################################
# Actor Loop (与原版相同)
##############################################################################


def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """Actor loop"""
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            import ipdb; ipdb.set_trace()
            done = False
            start_time = time.time()
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=False,
                    seed=key
                )
                actions = np.asarray(jax.device_get(actions))
                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs

                if done:
                    if reward:
                        dt = time.time() - start_time
                        time_list.append(dt)
                    success_counter += reward
                    print(reward)
                    print(f"{success_counter}/{episode + 1}")

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")
        return

    start_step = (
        int(os.path.basename(natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))[-1])[12:-4]) + 1
        if FLAGS.checkpoint_path and os.path.exists(os.path.join(FLAGS.checkpoint_path, "buffer"))
        and glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl"))
        else 0
    )

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    transitions = []
    demo_transitions = []
    obs, _ = env.reset()
    done = False

    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < config.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                )
                actions = np.asarray(jax.device_get(actions))

        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)
            if "left" in info:
                info.pop("left")
            if "right" in info:
                info.pop("right")

            if "intervene_action" in info:
                actions = info.pop("intervene_action")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False

            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            if 'grasp_penalty' in info:
                transition['grasp_penalty'] = info['grasp_penalty']
                print("grasp_penalty: ", info['grasp_penalty'])
            data_store.insert(transition)
            transitions.append(copy.deepcopy(transition))
            if already_intervened:
                intvn_data_store.insert(transition)
                demo_transitions.append(copy.deepcopy(transition))

            obs = next_obs
            if done or truncated:
                if reward > 0 or info.get("succeed", False):
                    print_green(f"✓ SUCCESS! Return: {running_return}")
                    
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                stats = {"environment": info}
                client.request("send-stats", stats)
                pbar.set_description(f"last return: {running_return}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                client.update()
                obs, _ = env.reset()

        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            if not os.path.exists(buffer_path):
                os.makedirs(buffer_path)
            if not os.path.exists(demo_buffer_path):
                os.makedirs(demo_buffer_path)
            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(transitions, f)
                transitions = []
            with open(os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(demo_transitions, f)
                demo_transitions = []

        timer.tock("total")

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################
# Learner Loop (与原版相同)
##############################################################################


def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
    """Learner loop"""
    start_step = (
        int(os.path.basename(checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)))[11:])
        + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        and checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        else 0
    )
    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}

    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)
    pbar.close()

    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    timer = Timer()
    
    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )

        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )


##############################################################################
# Main
##############################################################################


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)
    
    # 创建 SAC agent
    if config.setup_mode == 'single-arm-fixed-gripper' or config.setup_mode == 'dual-arm-fixed-gripper':   
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = False
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    elif config.setup_mode == 'dual-arm-learned-gripper':
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # replicate agent across devices
    agent = jax.device_put(
        jax.tree_map(jnp.array, agent), sharding.replicate()
    )

    # =====================================================================
    # 关键修改：从 BC checkpoint 初始化 actor 权重
    # =====================================================================
    bc_initialized = False
    
    if FLAGS.bc_checkpoint_path is not None and os.path.exists(FLAGS.bc_checkpoint_path):
        print_green("=" * 60)
        print_green("Initializing SAC actor from BC checkpoint")
        print_green("=" * 60)
        
        # 创建 BC agent 用于加载 checkpoint
        bc_agent: BCAgent = make_bc_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
        )
        bc_agent = jax.device_put(
            jax.tree_map(jnp.array, bc_agent), sharding.replicate()
        )
        
        # 迁移权重
        agent = transfer_bc_to_sac(bc_agent, agent, FLAGS.bc_checkpoint_path)
        bc_initialized = True
        print_green("=" * 60)
        print_green("BC -> SAC transfer complete!")
        print_green("=" * 60)
    
    # 如果有现有的 RL checkpoint，加载它
    rl_checkpoint_loaded = False
    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if latest_ckpt is not None:
            ckpt_step = int(os.path.basename(latest_ckpt)[11:])
            
            if ckpt_step == 0 and bc_initialized:
                # 这是 BC 初始化后的正常情况，不需要加载（当前 agent 已经是 BC 初始化的）
                print_green(f"Found BC-initialized checkpoint at step 0. Using current BC-initialized weights.")
            elif ckpt_step > 0:
                # 有训练过的 checkpoint，加载它（这会覆盖 BC 初始化）
                if bc_initialized:
                    print_yellow(f"Found trained checkpoint at step {ckpt_step}.")
                    user_input = input("Load trained checkpoint? This will OVERRIDE BC initialization. [y/N]: ")
                    if user_input.lower() != 'y':
                        print_yellow("Keeping BC-initialized weights. Will start from step 0.")
                        # 删除现有 checkpoint 目录或使用新目录
                        print_red("Warning: checkpoint_path already has trained checkpoints!")
                        print_red("Please use a different checkpoint_path or delete existing checkpoints.")
                        return
                
                ckpt = checkpoints.restore_checkpoint(
                    os.path.abspath(FLAGS.checkpoint_path),
                    agent.state,
                )
                agent = agent.replace(state=ckpt)
                rl_checkpoint_loaded = True
                print_green(f"Loaded RL checkpoint at step {ckpt_step}.")
            else:
                # step 0 但没有 BC 初始化，可能是之前失败的 run
                print_yellow(f"Found checkpoint at step 0 (not BC-initialized). Loading it.")
                ckpt = checkpoints.restore_checkpoint(
                    os.path.abspath(FLAGS.checkpoint_path),
                    agent.state,
                )
                agent = agent.replace(state=ckpt)
                rl_checkpoint_loaded = True
    
    # 如果是 BC 初始化且没有加载 RL checkpoint，保存初始 checkpoint
    if bc_initialized and not rl_checkpoint_loaded:
        print_yellow("Saving BC-initialized checkpoint as step 0...")
        os.makedirs(FLAGS.checkpoint_path, exist_ok=True)
        checkpoints.save_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path), agent.state, step=0, keep=100
        )
        print_green("Saved initial checkpoint from BC weights.")

    # 计算 start_step（用于 wandb 和训练）
    is_resuming = False
    start_step = 0
    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if latest_ckpt is not None:
            start_step = int(os.path.basename(latest_ckpt)[11:]) + 1
            is_resuming = start_step > 1  # step 0 是 BC 初始化的，不算 resume
    
    print_yellow(f"Training will start from step {start_step}")
    if is_resuming:
        print_yellow("This is a RESUME run")
    elif bc_initialized:
        print_yellow("This is a NEW run with BC initialization")
    else:
        print_yellow("This is a NEW run from scratch")

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        
        # 构建 wandb run name
        if is_resuming:
            run_name = f"{FLAGS.exp_name}_resume_from_{start_step}"
        elif bc_initialized:
            run_name = f"{FLAGS.exp_name}_bc_init"
        else:
            run_name = FLAGS.exp_name
        
        # 创建 wandb logger
        # 注意：如果需要真正的 resume，需要传入 run_id
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=run_name,
            debug=FLAGS.debug,
        )
        
        # 如果是 resume，在 wandb 中记录
        if wandb_logger and start_step > 0:
            wandb_logger.log({"resume_from_step": start_step}, step=start_step)
        
        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )

        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                        transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
            )

        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)
        intvn_data_store = QueuedDataStore(50000)

        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)