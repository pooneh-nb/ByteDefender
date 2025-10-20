"""Bianry for training bytecodes classifier transformer."""

import functools
import logging
from typing import Any, Callable, Sequence, Type

import chex
from clu import metric_writers
from clu import metrics
from clu import periodic_actions
import flax
import flax.linen as nn
from flax.training import train_state
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from orbax import checkpoint as orbax_checkpoint

import model
import seqio_utils


LearningRateSchedule = Callable[[chex.Numeric], chex.Numeric]


class Config(ml_collections.ConfigDict):
  """Configuration for training bytecodes classifier transformer."""

  model_name: str = ''
  seed: int = 42

  num_train_steps: int = 1_000
  initial_step: int = 1
  log_loss_every_steps = 25
  eval_every_steps = 250
  checkpoint_every_steps = 1_000

  learning_rate: float = 0.001
  lr_warmup_steps: int = 100
  lr_decay_steps: int = -1

  per_device_batch_size: int = 16
  train_tfrecord_path: str = ''
  test_tfrecord_path: str | None = None
  max_vocab_size: int = 256
  seqlen: int = 1024

  num_layers: int = 6
  num_heads: int = 4
  embed_dim: int = 128
  transformer_mlp_dim: int = 512
  conv_layers: int | None = None

  workdir: str = ''
  checkpoint_dir: str | None = None


class ClassifierConfig(Config):
  """Configuration for training the classifier model specifically."""

  classifier_mlp_dim: int = 128


def TrainClassifier(
    config: ClassifierConfig, allow_duplicate_tasks: bool = False
):
  """Trains a bytecodes classifier transformer."""
  logging.info('Training with config: %s', config)

  try:
    seqio_utils.AddClassifierTask(
        train_tfrecord_path=config.train_tfrecord_path,
        test_tfrecord_path=config.test_tfrecord_path
        if hasattr(config, 'test_tfrecord_path')
        else None,
        max_vocab_size=config.max_vocab_size,
        max_seqlen=config.seqlen,
    )
  except ValueError as e:  # Task already exists.
    # Allow duplicate tasks in Colab.
    if (
        allow_duplicate_tasks
        and 'Attempting to register duplicate provider' in str(e)
    ):
      pass
    else:
      raise e
  logging.info('Device count: %d', jax.device_count())
  global_batch_size = config.per_device_batch_size * jax.device_count()
  logging.info('Using global batch size: %d', global_batch_size)

  train_ds = seqio_utils.ClassifierTrainDataset(global_batch_size, config.seed)
  train_ds = train_ds.shard(
      num_shards=jax.process_count(), index=jax.process_index()
  )
  train_ds_iter = iter(train_ds)

  if (
      hasattr(config, 'test_tfrecord_path')
      and config.test_tfrecord_path is not None
  ):
    test_ds = seqio_utils.ClassifierTestDataset(global_batch_size, config.seed)
    test_ds = test_ds.shard(
        num_shards=jax.process_count(), index=jax.process_index()
    )
  else:
    test_ds = None

  mesh, data_sharding, data_spec, var_sharding, var_spec = _CreateMesh()

  # Learning rate schedule.
  learning_rate_fn = functools.partial(LearningRate, config=config)

  # Create train state.
  logging.info('Using seed: %d', config.seed)
  rng = jax.random.PRNGKey(config.seed)
  state = _CreateTrainState(
      model.Classifier(
          vocab_size=config.max_vocab_size,
          embed_dim=config.embed_dim,
          seqlen=config.seqlen,
          num_layers=config.num_layers,
          num_heads=config.num_heads,
          tfrmr_hidden_dim=config.transformer_mlp_dim,
          cls_hidden_dim=config.classifier_mlp_dim,
          conv_layers=(
              config.conv_layers if hasattr(config, 'conv_layers') else None
          ),
      ),
      rng,
      input_shape=(config.per_device_batch_size, config.seqlen),
      learning_rate_fn=learning_rate_fn,
      initial_step=config.initial_step,
  )
  initial_step = int(state.step)
  logging.info(
      'Total parameters: %d',
      sum(np.prod(x.shape) for x in jax.tree.leaves(state.params))
  )

  checkpoint_manager, state = _CreateCheckpointManager(config, state)

  train_step = _CreateParallelStep(
      functools.partial(
          _TrainStep,
          learning_rate_fn=learning_rate_fn,
          compute_loss_fn=_BinaryCrossEntropyLoss,
          metrics_cls=ClassifierTrainMetrics,
      ),
      mesh,
      data_sharding,
      data_spec,
      var_sharding,
      var_spec,
  )
  eval_step = _CreateParallelStep(
      _ClassifierEvalStep,
      mesh,
      data_sharding,
      data_spec,
      var_sharding,
      var_spec,
  )

  train_metrics = None
  writer = metric_writers.create_default_writer(
      config.workdir,
      just_logging=jax.process_index() > 0,
  )
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=config.num_train_steps, writer=writer
  )
  hooks = []
  if jax.process_index() == 0:
    hooks.append(report_progress)

  for step in range(initial_step, config.num_train_steps + 1):
    assert step == state.step
    if step == 1:
      writer.write_hparams(dict(config))
    with mesh:
      with jax.profiler.StepTraceAnnotation('train', step_num=step):
        # Cycle training data for as long as it takes to complete training
        # steps.
        try:
          batch = next(train_ds_iter)
        except StopIteration:
          train_ds_iter = iter(train_ds)
          batch = next(train_ds_iter)

        # logging.info('No. FP examples: %s', np.sum(batch['label']))
        x = jnp.array(batch['bytecodes'])
        y = jnp.array(batch['label'])
        state, metrics_update = train_step(state, x, y)
      train_metrics = (
          metrics_update
          if train_metrics is None
          else train_metrics.merge(metrics_update)
      )

      for h in hooks:
        h(step)

      # Checkpoint.
      if (
          hasattr(config, 'checkpoint_dir')
          and config.checkpoint_dir is not None
      ) and (
          step % config.checkpoint_every_steps == 0
          or step == config.num_train_steps
      ):
        assert checkpoint_manager is not None
        with report_progress.timed('checkpoint'):
          checkpoint_manager.save(
              step,
              items=dict(
                  train_state=jax.tree.map(np.array, state),
              ),
          )
      # Flush train metrics.
      if (
          step % config.log_loss_every_steps == 0
          or step == config.num_train_steps
      ):
        writer.write_scalars(step, train_metrics.compute())
        train_metrics = None
      # Evaluate on test set.
      if (
          hasattr(config, 'test_tfrecord_path')
          and config.test_tfrecord_path is not None
      ) and (
          step % config.eval_every_steps == 0 or step == config.num_train_steps
      ):
        assert test_ds is not None
        with report_progress.timed('eval'):
          eval_metrics = None
          for batch in iter(test_ds):
            x = jnp.array(batch['bytecodes'])
            y = jnp.array(batch['label'])
            _, metrics_update = eval_step(state, x, y)
            eval_metrics = (
                metrics_update
                if eval_metrics is None
                else eval_metrics.merge(metrics_update)
            )
        if eval_metrics is not None:
          writer.write_scalars(step, eval_metrics.compute())
  writer.flush()


def _CreateMesh() -> tuple[
    jax.sharding.Mesh,
    jax.sharding.NamedSharding,
    jax.sharding.PartitionSpec,
    jax.sharding.NamedSharding,
    jax.sharding.PartitionSpec,
]:
  """Create device mesh for data sharding."""
  # Use multiple devices if available for data parallelism.
  device_mesh = mesh_utils.create_device_mesh(
      mesh_shape=(jax.device_count(),),
      devices=jax.devices(),
      allow_split_physical_axes=True,
  )
  mesh = jax.sharding.Mesh(device_mesh, ('devices',))
  # Shards the batch axis across all devices.
  data_spec = jax.sharding.PartitionSpec('devices')
  data_sharding = jax.sharding.NamedSharding(mesh, data_spec)
  # The model parameters are not sharded, instead they are replicated across
  # all devices and gradients are averaged across all devices.
  var_spec = jax.sharding.PartitionSpec()
  var_sharding = jax.sharding.NamedSharding(mesh, var_spec)
  return mesh, data_sharding, data_spec, var_sharding, var_spec


class TrainState(train_state.TrainState):
  step: int
  params: Any
  opt_state: optax.OptState


def _CreateCheckpointManager(
    config: Config,
    state: TrainState,
) -> tuple[orbax_checkpoint.CheckpointManager | None, TrainState]:
  """Create a checkpoint manager for the training state."""
  if not hasattr(config, 'checkpoint_dir') or config.checkpoint_dir is None:
    return None, state
  checkpointers = dict(train_state=orbax_checkpoint.PyTreeCheckpointer())
  checkpoint_manager = orbax_checkpoint.CheckpointManager(
      config.checkpoint_dir,
      checkpointers=checkpointers,
      options=orbax_checkpoint.CheckpointManagerOptions(create=True),
  )
  checkpoint_state = dict(train_state=state)
  if checkpoint_manager.latest_step() is not None:
    state = checkpoint_manager.restore(
        checkpoint_manager.latest_step(), items=checkpoint_state
    )
  return checkpoint_manager, state


def _GetPredictedLabels(logits: jnp.ndarray) -> jnp.ndarray:
  if logits.shape[-1] == 1:
    return (logits.squeeze(-1) >= 0.5).astype(jnp.float32)
  return jnp.argmax(logits, axis=-1, keepdims=False).astype(jnp.float32)


@flax.struct.dataclass
class ClassifierAccuracy(metrics.Accuracy):

  @classmethod
  def from_model_output(
      cls, *, logits: jnp.ndarray, labels: jnp.ndarray, **kwargs
  ):
    metric = metrics.Average.from_model_output(
        values=(_GetPredictedLabels(logits) == labels).astype(jnp.float32),
        **kwargs,
    )
    return cls(**vars(metric))  # cls(metrics) doesn't work for a dataclass


@flax.struct.dataclass
class Recall(metrics.Metric):
  """Custom binary recall metric."""
  confusion_matrix: jnp.ndarray

  @classmethod
  def from_model_output(cls, *, logits: jnp.ndarray, labels: jnp.ndarray, **_):
    return cls(
        confusion_matrix=_ConfusionMatrix(labels, _GetPredictedLabels(logits)),
    )

  def compute(self):
    return super().compute()[1]

  def merge(self, other: 'Recall') -> 'Recall':
    return type(self)(
        confusion_matrix=self.confusion_matrix + other.confusion_matrix)

  def compute(self):
    true_positives = jnp.diag(self.confusion_matrix)
    denominator = jnp.sum(self.confusion_matrix, axis=1)
    precision = _DivideNoNaN(true_positives, denominator)
    return precision[1]


def _ConfusionMatrix(labels: jnp.ndarray, logits: jnp.ndarray) -> jnp.ndarray:
  return jnp.histogram2d(
      labels.ravel(),
      logits.ravel(),
      bins=jnp.arange(3),
  )[0]

def _DivideNoNaN(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  dtype = jnp.result_type(x, y)
  y_is_zero = jnp.equal(y, 0.)
  div = jnp.divide(x, jnp.where(y_is_zero, jnp.ones((), dtype=dtype), y))
  return jnp.where(y_is_zero, jnp.zeros((), dtype=dtype), div)


@flax.struct.dataclass
class Precision(metrics.Metric):
  """Custom binary precision metric."""
  confusion_matrix: jnp.ndarray

  @classmethod
  def from_model_output(cls, *, logits: jnp.ndarray, labels: jnp.ndarray, **_):
    return cls(
        confusion_matrix=_ConfusionMatrix(labels, _GetPredictedLabels(logits)),
    )

  def merge(self, other: 'Precision') -> 'Precision':
    return type(self)(
        confusion_matrix=self.confusion_matrix + other.confusion_matrix)

  def compute(self):
    true_positives = jnp.diag(self.confusion_matrix)
    false_positives = jnp.sum(self.confusion_matrix, axis=0) - true_positives
    precision = _DivideNoNaN(true_positives, true_positives + false_positives)
    return precision[1]


@flax.struct.dataclass
class ClassifierTrainMetrics(metrics.Collection):
  learning_rate: metrics.LastValue.from_output('learning_rate')
  train_loss: metrics.Average.from_output('loss')
  train_std: metrics.Std.from_output('loss')
  train_accuracy: ClassifierAccuracy
  train_recall: Recall
  train_precision: Precision


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
  eval_accuracy: ClassifierAccuracy
  eval_recall: Recall
  eval_precision: Precision
  eval_loss: metrics.Average.from_output('loss')
  eval_std: metrics.Std.from_output('loss')


def _CreateTrainState(
    mdl: nn.Module,
    rng: jnp.ndarray,
    input_shape: Sequence[int],
    learning_rate_fn: LearningRateSchedule,
    initial_step: int,
) -> TrainState:
  """Initializes the model, optimizer, and train state."""
  variables = mdl.init(rng, np.zeros(input_shape, dtype=np.int64))
  params = variables['params']
  tx = optax.adam(learning_rate=learning_rate_fn)
  state = TrainState.create(
      apply_fn=mdl.apply,
      params=params,
      tx=tx,
  )
  state = state.replace(step=initial_step)
  return state


def LearningRate(step: int, *, config: Config) -> float | jnp.ndarray:
  warmup_steps = jnp.maximum(config.lr_warmup_steps, 1)
  warmup = jnp.minimum(1.0, step / warmup_steps)
  # Cosine decay.
  ratio = jnp.maximum(0.0, step - config.lr_warmup_steps)
  decay_steps = jax.lax.cond(
      config.lr_decay_steps > 0,
      lambda: float(config.lr_decay_steps),
      lambda: float('inf'),
  )
  ratio /= jnp.maximum(1.0, decay_steps)
  mult = 0.5 * (1.0 + jnp.cos(jnp.pi * ratio))
  return warmup * mult * config.learning_rate


def _CreateParallelStep(
    train_step_fn: Callable,  # pylint: disable=g-bare-generic
    mesh: jax.sharding.Mesh,
    data_sharding: jax.sharding.NamedSharding,
    data_spec: jax.sharding.PartitionSpec,
    var_sharding: jax.sharding.NamedSharding,
    var_spec: jax.sharding.PartitionSpec,
) -> Callable:  # pylint: disable=g-bare-generic
  """Create the train step function for data parallelism."""
  train_step = jax.experimental.shard_map.shard_map(
      train_step_fn,
      mesh=mesh,
      in_specs=(var_spec, data_spec, data_spec),
      out_specs=(var_spec, var_spec),
      check_rep=False,
  )
  train_step = jax.jit(
      train_step,
      in_shardings=(var_sharding, data_sharding, data_sharding),
      out_shardings=(var_sharding, var_sharding),
  )
  return train_step


def _TrainStep(
    state: TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    compute_loss_fn: Callable,  # pylint: disable=g-bare-generic
    learning_rate_fn: LearningRateSchedule,
    metrics_cls: Type[ClassifierTrainMetrics],
) -> tuple[TrainState, ClassifierTrainMetrics]:
  """Single train step computes loss, gradients, and updates parameters."""
  def _LossFn(params: Any) -> tuple[jnp.ndarray, Any]:
    logits, _ = state.apply_fn({'params': params}, x, mutable=[])
    return compute_loss_fn(logits, y), logits

  (loss, logits), grads = jax.value_and_grad(_LossFn, has_aux=True)(
      state.params
  )
  grad = jax.lax.psum(grads, axis_name='devices')
  new_state = state.apply_gradients(grads=grad)
  if len(y.shape) > 1:
    if y.shape[-1] > 1:
      labels = jnp.argmax(y, axis=-1, keepdims=False)
    else:
      labels = y.squeeze(-1)
  else:
    labels = y
  metrics_update = metrics_cls.gather_from_model_output(
      axis_name='devices',
      labels=labels,
      logits=logits,
      loss=loss,
      learning_rate=learning_rate_fn(new_state.step),
  )
  return new_state, metrics_update


def _BinaryCrossEntropyLoss(
    logits: jnp.ndarray, labels: jnp.ndarray
) -> jnp.ndarray:
  logits = jnp.clip(logits, 1e-12, 1.0 - 1e-12)
  binary_x_entropy = -jnp.log(logits) * labels
  binary_x_entropy -= jnp.log(1.0 - logits) * (1.0 - labels)
  return jnp.sum(binary_x_entropy)


def _ClassifierEvalStep(
    state: TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> EvalMetrics:
  """Single eval step computes loss and metrics."""
  logits, _ = state.apply_fn({'params': state.params}, x, mutable=[])
  loss = _BinaryCrossEntropyLoss(logits, y)
  return state, EvalMetrics.gather_from_model_output(
      axis_name='devices',
      labels=jnp.squeeze(y, -1),
      logits=logits,
      loss=loss,
  )
