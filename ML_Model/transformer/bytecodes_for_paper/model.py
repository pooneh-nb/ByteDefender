"""Implementation of a bytecode classifier transformer."""

import math

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


PAD_TOKEN_ID = 0


class MLP(nn.Module):
  """Multi-layer perceptron."""

  hidden_dim: int
  output_dim: int

  def setup(self):
    self.k = nn.Dense(self.hidden_dim)
    self.v = nn.Dense(self.output_dim)

  def __call__(self, x: jnp.ndarray):
    x = self.k(x)
    x = nn.relu(x)
    return self.v(x)


class TransformerLayer(nn.Module):
  """Transformer block."""

  embed_dim: int
  num_heads: int
  hidden_dim: int

  def setup(self):
    self.attn = nn.attention.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        qkv_features=self.embed_dim,
    )
    self.norm1 = nn.LayerNorm(epsilon=1e-6)
    self.mlp = MLP(self.hidden_dim, self.embed_dim)
    self.norm2 = nn.LayerNorm(epsilon=1e-6)

  def __call__(self, x: jnp.ndarray, mask: jnp.ndarray):
    y = self.attn(x, mask=mask)
    x = self.norm1(x + y)
    z = self.mlp(x)
    return self.norm2(x + z)


class PositionalEncoding(nn.Module):
  """Based on https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html"""
  embed_dim: int
  seqlen: int

  def setup(self):
    pe = np.zeros((self.seqlen, self.embed_dim))
    position = np.arange(0, self.seqlen, dtype=jnp.float32)[:, None]
    div_term = np.exp(
        np.arange(0, self.embed_dim, 2) * (-math.log(10000.0) / self.embed_dim)
    )
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = pe[None]
    self.pe = jax.device_put(pe)

  def __call__(self, x):
    return self.pe[:, :x.shape[1]]


class Transformer(nn.Module):
  """Transformer network with position and token embeddings."""

  vocab_size: int
  embed_dim: int
  seqlen: int
  num_layers: int
  num_heads: int
  hidden_dim: int
  causal: bool
  conv_layers: int | None = None

  def setup(self):
    self.embed = nn.Embed(self.vocab_size, self.embed_dim)
    pos_embed_len = self.seqlen
    if self.conv_layers is not None:
      # Downsample along sequence dimension by factor of 2 per conv block.
      # Apply poositional encoding after downsampling.
      pos_embed_len = pos_embed_len // (2 ** self.conv_layers)
    self.pos_embed = PositionalEncoding(self.embed_dim, pos_embed_len)
    self.layers = [
        TransformerLayer(self.embed_dim, self.num_heads, self.hidden_dim)
        for _ in range(self.num_layers)
    ]
    if self.conv_layers is not None:
      # Downsample along sequence dimension by factor of 2 per conv block.
      self.conv_block_lyrs = [
          nn.Conv(
              features=self.embed_dim,
              kernel_size=2,
              strides=2,
              padding='SAME',
              use_bias=False,
          )
          for _ in range(self.conv_layers)
      ]

  def __call__(self, tokens: jnp.ndarray):
    x = self.embed(tokens)
    if self.causal:
      mask = CreateCausalMask(tokens, self.num_heads)
    else:
      mask = CreatePaddingMask(tokens, self.num_heads)
    if self.conv_layers is not None:
      for conv_block in self.conv_block_lyrs:
        mask = DownsampleAttentionMask(mask)
        x = conv_block(x)
    x += self.pos_embed(x)
    for lyr in self.layers:
      x = lyr(x, mask)
    return x


def CreateCausalMask(tokens: jnp.ndarray, n_heads: int) -> jnp.ndarray:
  """Creates a causal attention mask for the given input."""
  assert tokens.ndim == 2  # (B, T)
  padding_mask = CreatePaddingMask(tokens, n_heads)  # (B, H, T, T)
  causal_mask = jnp.tril(jnp.ones_like(padding_mask), k=0)  # (B, H, T, T)
  return jnp.logical_and(padding_mask, causal_mask)


def CreatePaddingMask(tokens: jnp.ndarray, n_heads: int) -> jnp.ndarray:
  """Creates an attention mask for the given input."""
  assert tokens.ndim == 2  # (B, T)
  mask = jnp.not_equal(tokens, PAD_TOKEN_ID)  # (B, T)
  assert mask.ndim == 2
  padding_mask = mask
  mask = jnp.expand_dims(mask, axis=1)  # (B, 1, T)
  mask = jnp.expand_dims(mask, axis=1)  # (B, 1, 1, T)
  mask = jnp.tile(mask, [1, n_heads, mask.shape[-1], 1])  # (B, H, T, T)
  mask = jnp.einsum('bi,bhij->bhij', padding_mask, mask)
  return mask


def DownsampleAttentionMask(mask: jnp.ndarray) -> jnp.ndarray:
  """Downsamples the mask by factor of 2 along the sequence dimension."""
  assert mask.ndim == 4  # (B, H, T, T)
  mask = nn.max_pool(
      jnp.expand_dims(mask, -1), window_shape=(1, 2, 2), strides=(1, 2, 2)
  )
  return mask.squeeze(-1)


class Classifier(nn.Module):
  """Bytecode classifier with transformer global avg pooling at the head."""

  vocab_size: int
  embed_dim: int
  seqlen: int
  num_layers: int
  num_heads: int
  tfrmr_hidden_dim: int
  cls_hidden_dim: int
  conv_layers: int | None = None

  def setup(self):
    self.transformer = Transformer(
        self.vocab_size,
        self.embed_dim,
        self.seqlen,
        self.num_layers,
        self.num_heads,
        self.tfrmr_hidden_dim,
        causal=False,
        conv_layers=self.conv_layers,
    )
    self.mlp = MLP(self.cls_hidden_dim, 1)

  def __call__(self, tokens: jnp.ndarray):
    x = self.transformer(tokens)  # (B, T, D)
    padding_mask = jnp.expand_dims(tokens != PAD_TOKEN_ID, axis=-1)
    if self.conv_layers is not None:
      for _ in range(self.conv_layers):
        padding_mask = nn.max_pool(
            padding_mask, window_shape=(2,), strides=(2,)
        )
    x *= padding_mask  # (B, T, D)

    # Take mean of non-padding tokens.
    x = jnp.sum(x, axis=-2, keepdims=False)  # (B, D)
    denom = jnp.sum(padding_mask, axis=-2, keepdims=False)  # (B, D)
    x /= denom  # (B, D)

    x = self.mlp(x)  # (B, 1)
    return nn.sigmoid(x)  # (B, 1)


class AutoregressiveTransformer(nn.Module):
  """Autoregressive transformer network."""
  vocab_size: int
  embed_dim: int
  seqlen: int
  num_layers: int
  num_heads: int
  hidden_dim: int

  def setup(self):
    self.transformer = Transformer(
        self.vocab_size,
        self.embed_dim,
        self.seqlen,
        self.num_layers,
        self.num_heads,
        self.hidden_dim,
        causal=True,
    )

  def __call__(self, tokens: jnp.ndarray):
    x = self.transformer(tokens)  # (B, T, D)
    # Use the embedding matrix as the final projection matrix.
    # This gives the embeddings more signal from training.
    x = jnp.einsum(
        '...i,ji->...j', x, self.transformer.embed.embedding
    )  # (B, T, V)
    return nn.softmax(x, axis=-1)
