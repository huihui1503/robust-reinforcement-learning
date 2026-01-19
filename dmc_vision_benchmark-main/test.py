import tensorflow_datasets as tfds
from typing import Any, List

import rlds
from rlds import rlds_types
import tensorflow as tf

@tf.function
def pad_and_crop_episode(
    episode: dict[str, Any], episode_length: int
) -> dict[str, Any]:
  """Returns a new episode where obs, actions, rewards... are of episode_length.

  Pad all the sequences in an episode to be of episode_length.

  Args:
    episode:  Incoming episode
    episode_length:  Length of windows to use
  """
  result = {}

  def pad_and_crop_all_leaves(ep, res):
    """Recursive function to pad and crop the leaves of a tree."""
    for key, value in ep.items():
      res[key] = {}
      if isinstance(value, dict):
        pad_and_crop_all_leaves(value, res[key])
      else:
        if len(value.shape) >= 2:
          # Overwrite the first entry of tf.shape(value)
          padded = tf.zeros(
              tf.tensor_scatter_nd_update(
                  tf.shape(value),
                  [[0]],
                  [episode_length - tf.shape(value)[0]],
              ),
              dtype=value.dtype,
          )
        else:
          padded = tf.zeros(
              episode_length - tf.shape(value)[0], dtype=value.dtype
          )

        # Concatenate the original values with the padded ones
        concat = tf.concat([value, padded], axis=0)

        # Guarantee the shapes returned - required since drop_remainder=False
        res[key] = tf.ensure_shape(concat, (episode_length,) + value.shape[1:])

  pad_and_crop_all_leaves(episode, result)
  return result

def episode_steps_to_batched_transition(
    episode: dict[str, Any],
    episode_length: int,
    only_first_window: bool,
    shift: int,
    drop_remainder: bool = False,
) -> dict[str, Any]:
  """Returns a new episode as a series of overlapping windows of steps.

  Also adds a timestep.

  Args:
    episode:  Incoming episode
    episode_length:  Length of windows to use
    only_first_window:  Whether to only return first window, as done in Raparthy
      et al.
    shift:  Increment to compute index to select the next element of each batch
    drop_remainder: Whether to drop the last few steps, or keep all of them and
      create batches padded with dummy observations.
  """
  new_episode = dict(episode)

  # Set the batch to episode_length.
  new_episode[rlds.STEPS] = rlds.transformations.batch(
      new_episode[rlds.STEPS],
      shift=shift,
      size=episode_length,
      drop_remainder=drop_remainder,
  )

  # For each episode, only take first episode_length steps.
  if only_first_window:
    new_episode[rlds.STEPS] = new_episode[rlds.STEPS].take(1)

  # Pad and crop each episode
  new_episode[rlds.STEPS] = new_episode[rlds.STEPS].map(
      lambda x: pad_and_crop_episode(x, episode_length)
  )
  return new_episode

data_dir = '/scratch/cor54gyp/dmc_vision_bench_data/dmc_vision_benchmark/locomotion/walker_walk/expert/none'
dataset_builder = tfds.builder_from_directories([data_dir])
ds = dataset_builder.as_dataset(
    split='train[:95%]',
    shuffle_files=True,
)

size = ds.cardinality().numpy()
print(f"size {size}") # 20 x 100 x 95% = 1900

# Take only a few samples to avoid a massive output
for example in ds.take(1):
    # 'example' is a dictionary
    tmp = episode_steps_to_batched_transition(
            episode=example,
            episode_length=3,
            only_first_window=False,
            shift=1,
            drop_remainder=False,
        )[rlds.STEPS]
    for step in tmp:
        action = step["action"]
        discount = step["discount"]
        is_first = step["is_first"]
        is_last = step["is_last"]
        is_terminal = step["is_terminal"]
        observation = step["observation"]
        reward = step["reward"]
        print(f" action {action}")
        print(f" discount {discount}")
        print(f" is_first {is_first}")
        print(f" is_last {is_last}")
        print(f" is_terminal {is_terminal}")
        print(f" observation {observation}")
        print(f" reward {reward}")
        break