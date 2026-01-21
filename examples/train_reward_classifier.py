import glob
import os
import pickle as pkl
import random
import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training import checkpoints
import numpy as np
import optax
from tqdm import tqdm
from absl import app, flags

from serl_launcher.data.data_store import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier

from experiments.mappings import CONFIG_MAPPING


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 150, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
# 新增：指定数据路径
flags.DEFINE_string("positive_path", None, "Path to success pkl file (optional, default: auto glob)")
flags.DEFINE_string("negative_path", None, "Path to failure pkl file (optional, default: auto glob)")
# 新增：数据平衡
flags.DEFINE_bool("balance_data", True, "Whether to balance positive and negative samples")
flags.DEFINE_float("neg_ratio", 100.0, "Ratio of negative to positive samples (e.g., 2.0 means 2x negatives)")


def load_transitions(paths, label):
    """加载并标记 transitions"""
    transitions = []
    for path in paths:
        data = pkl.load(open(path, "rb"))
        for trans in data:
            if "images" in trans['observations'].keys():
                continue
            trans["labels"] = label
            transitions.append(trans)
    return transitions


def weighted_bce_with_logits(logits, labels, pos_weight):
    # labels: {0,1}
    loss = optax.sigmoid_binary_cross_entropy(logits, labels)
    weights = labels * pos_weight + (1.0 - labels)
    return loss * weights


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)

    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)
    
    # 获取数据路径
    if FLAGS.positive_path:
        success_paths = [FLAGS.positive_path]
    else:
        success_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*success*.pkl"))
    
    if FLAGS.negative_path:
        failure_paths = [FLAGS.negative_path]
    else:
        failure_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*failure*.pkl"))
    
    print(f"Success paths: {success_paths}")
    print(f"Failure paths: {failure_paths}")
    
    # 加载数据
    pos_transitions = load_transitions(success_paths, label=1)
    neg_transitions = load_transitions(failure_paths, label=0)
    
    print(f"\n原始数据量:")
    print(f"  Success: {len(pos_transitions)}")
    print(f"  Failure: {len(neg_transitions)}")
    
    pos_weight = len(neg_transitions) * 1.0 / len(pos_transitions)
    print(f"  pos_weight: {pos_weight}")
    
    
    
    # # 数据平衡
    # if FLAGS.balance_data and len(neg_transitions) > len(pos_transitions) * FLAGS.neg_ratio:
    #     target_neg = int(len(pos_transitions) * FLAGS.neg_ratio)
    #     neg_transitions = random.sample(neg_transitions, target_neg)
    #     print(f"\n平衡后数据量:")
    #     print(f"  Success: {len(pos_transitions)}")
    #     print(f"  Failure: {len(neg_transitions)} (下采样到 {FLAGS.neg_ratio}x)")
    
    # Create buffer for positive transitions
    pos_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=20000,
        include_label=True,
    )
    for trans in pos_transitions:
        trans['actions'] = env.action_space.sample()
        pos_buffer.insert(trans)
            
    pos_iterator = pos_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=sharding.replicate(),
    )
    
    # Create buffer for negative transitions
    neg_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=50000,
        include_label=True,
    )
    for trans in neg_transitions:
        trans['actions'] = env.action_space.sample()
        neg_buffer.insert(trans)
            
    neg_iterator = neg_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=sharding.replicate(),
    )

    print(f"\nBuffer sizes:")
    print(f"  pos_buffer: {len(pos_buffer)}")
    print(f"  neg_buffer: {len(neg_buffer)}")

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    pos_sample = next(pos_iterator)
    neg_sample = next(neg_iterator)
    sample = concat_batches(pos_sample, neg_sample, axis=0)

    rng, key = jax.random.split(rng)
    classifier = create_classifier(key, 
                                   sample["observations"], 
                                   config.classifier_keys,
                                   )

    def data_augmentation_fn(rng, observations):
        for pixel_key in config.classifier_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    @jax.jit
    def train_step(state, batch, key):
        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params}, batch["observations"], rngs={"dropout": key}, train=True
            )
            # return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

            loss = weighted_bce_with_logits(
                logits,
                batch["labels"],
                pos_weight,
            )
            return loss.mean()

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        logits = state.apply_fn(
            {"params": state.params}, batch["observations"], train=False, rngs={"dropout": key}
        )
        train_accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])

        return state.apply_gradients(grads=grads), loss, train_accuracy

    print("\n开始训练...")
    for epoch in tqdm(range(FLAGS.num_epochs)):
        # Sample equal number of positive and negative examples
        pos_sample = next(pos_iterator)
        neg_sample = next(neg_iterator)
        # Merge and create labels
        batch = concat_batches(
            pos_sample, neg_sample, axis=0
        )
        rng, key = jax.random.split(rng)
        obs = data_augmentation_fn(key, batch["observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "labels": batch["labels"][..., None],
            }
        )
            
        rng, key = jax.random.split(rng)
        classifier, train_loss, train_accuracy = train_step(classifier, batch, key)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
            )

    ckpt_dir = os.path.join(os.getcwd(), "classifier_ckpt/")
    checkpoints.save_checkpoint(
        ckpt_dir,
        classifier,
        step=FLAGS.num_epochs,
        overwrite=True,
    )
    print(f"\n模型已保存到: {ckpt_dir}")
    

if __name__ == "__main__":
    app.run(main)