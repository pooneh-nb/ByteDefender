"""Code for handling interactions with SeqIO library."""

from collections.abc import Mapping
import functools

import jax
import seqio
import tensorflow as tf

import model


CLASSIFIER_TASK_NAME = 'bytecodes_classifier'
PRETRAIN_TASK_NAME = 'bytecodes_pretrain'


def AddClassifierTask(
    train_tfrecord_path: str,
    max_vocab_size: int,
    max_seqlen: int,
    test_tfrecord_path: str | None = None,
):
  return _AddTaskInternal(
      train_tfrecord_path,
      max_vocab_size,
      max_seqlen,
      test_tfrecord_path,
      include_label=True,
      task_name=CLASSIFIER_TASK_NAME,
  )


def AddPretrainTransformerTask(
    train_tfrecord_path: str,
    max_vocab_size: int,
    max_seqlen: int,
    test_tfrecord_path: str | None = None,
):
  return _AddTaskInternal(
      train_tfrecord_path,
      max_vocab_size,
      max_seqlen,
      test_tfrecord_path,
      include_label=False,
      task_name=PRETRAIN_TASK_NAME,
  )


def _AddTaskInternal(
    train_tfrecord_path: str,
    max_vocab_size: int,
    max_seqlen: int,
    test_tfrecord_path: str | None,
    include_label: bool,
    task_name: str,
):
  """Add a SeqIO task for the bytecode classifier when given hyperparameters."""
  split_to_filepattern = {
      'train': train_tfrecord_path,
  }
  if test_tfrecord_path:
    split_to_filepattern['test'] = test_tfrecord_path

  feature_description = {
      'bytecodes': tf.io.FixedLenSequenceFeature(
          shape=[],
          dtype=tf.int64,
          default_value=model.PAD_TOKEN_ID,
          allow_missing=True,
      ),
  }
  output_features = {
      'bytecodes': seqio.Feature(
          vocabulary=seqio.PassThroughVocabulary(size=max_vocab_size),
          add_eos=False,
          required=True,
          dtype=tf.int64,
      ),
  }
  if include_label:
    feature_description['label'] = tf.io.FixedLenFeature(
        shape=[1],
        dtype=tf.int64,
    )
    output_features['label'] = seqio.Feature(
        vocabulary=seqio.PassThroughVocabulary(size=2),
        add_eos=False,
        required=True,
        dtype=tf.int64,
    )

  seqio.TaskRegistry.add(
      task_name,
      seqio.TFExampleDataSource(
          split_to_filepattern=split_to_filepattern,
          feature_description=feature_description,
      ),
      output_features=output_features,
      preprocessors=[
          functools.partial(_EnforceMaxLength, max_seqlen=max_seqlen)
      ],
  )


def _EnforceMaxLength(ds: tf.data.Dataset, max_seqlen: int):
  def _Internal(ex: Mapping[str, tf.Tensor]):
    return jax.tree.map(lambda x: x[:max_seqlen], ex)

  return ds.map(_Internal, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def ClassifierTrainDataset(
    global_batch_size: int, seed: int
) -> tf.data.Dataset:
  return _DatasetInternal(
      CLASSIFIER_TASK_NAME, 'train', global_batch_size, seed
  )


def ClassifierTestDataset(global_batch_size: int, seed: int) -> tf.data.Dataset:
  return _DatasetInternal(CLASSIFIER_TASK_NAME, 'test', global_batch_size, seed)


def PretrainTrainDataset(global_batch_size: int, seed: int) -> tf.data.Dataset:
  return _DatasetInternal(PRETRAIN_TASK_NAME, 'train', global_batch_size, seed)


def PretrainTestDataset(global_batch_size: int, seed: int) -> tf.data.Dataset:
  return _DatasetInternal(PRETRAIN_TASK_NAME, 'test', global_batch_size, seed)


def _DatasetInternal(
    task_name: str, split: str, global_batch_size: int, seed: int
) -> tf.data.Dataset:
  return (
      seqio.TaskRegistry.get(task_name)
      .get_dataset(
          split=split,
          shuffle=True,
          seed=seed,
          use_cached=False,
      )
      .padded_batch(global_batch_size, drop_remainder=True)
  )
