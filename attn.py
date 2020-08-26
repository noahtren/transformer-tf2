import code
from typing import Dict

import tensorflow as tf


class Boom(tf.keras.layers.Layer):
  """Boom layer. Naming inspired by Stephen Merity.
  This layer is used in the original Transformer (Vaswani et. al, 2017)
  """
  def __init__(self, config, **kwargs):
    super(Boom, self).__init__(**kwargs)
    self.config = config
    self.attrs = ['dim']
    for attr in self.attrs:
      assert attr in config, f"`{attr}` must be in config of `{self.__class__.__name__}` layer"
      setattr(self, attr, config[attr])
    self.w1 = tf.keras.layers.Dense(self.dim * 4, name=f'boom_w1_{self.name}')
    self.w2 = tf.keras.layers.Dense(self.dim, name=f'boom_w2_{self.name}')


  def call(self, inputs):
    x = inputs
    x = self.w1(x)
    x = tf.nn.gelu(x, name=f'boom_gelu_{self.name}')
    x = self.w2(x)
    return x


  def get_config(self):
    config = super(Boom, self).get_config()
    config.update(self.config)
    return config


class MultiHeadSelfAttention(tf.keras.layers.Layer):
  """Multi-head self-attention
  """
  def __init__(self, config, **kwargs):
    super(MultiHeadSelfAttention, self).__init__(**kwargs)
    # config
    self.config = config
    self.attrs = ['dk', 'dv', 'seq_len', 'num_heads']
    for attr in self.attrs:
      assert attr in config, f"`{attr}` must be in config of `{self.__class__.__name__}` layer"
      setattr(self, attr, config[attr])

    # init
    self.q_w = tf.keras.layers.Dense(self.num_heads * self.dk, name=f'q_proj_{self.name}')
    self.k_w = tf.keras.layers.Dense(self.num_heads * self.dk, name=f'k_proj_{self.name}')
    self.v_w = tf.keras.layers.Dense(self.num_heads * self.dv, name=f'v_proj_{self.name}')
    self.linear = tf.keras.layers.Dense(self.dv, name=f'mh_out_proj_{self.name}')


  def call(self, inputs):
    """Inputs:
      x: [batch_size, seq_len, dv]
    """
    x = inputs
    # linear
    q_raw = self.q_w(x)
    k_raw = self.k_w(x)
    v_raw = self.v_w(x)

    q_embeds = tf.reshape(q_raw, [-1, self.seq_len, self.num_heads, self.dk], name=f'q_embed_{self.name}')
    k_embeds = tf.reshape(k_raw, [-1, self.seq_len, self.num_heads, self.dk], name=f'k_embed_{self.name}')
    v_embeds = tf.reshape(v_raw, [-1, self.seq_len, self.num_heads, self.dv], name=f'v_embed_{self.name}')

    q_embeds = tf.transpose(q_embeds, [0, 2, 1, 3], name=f'q_permute_{self.name}')
    k_embeds = tf.transpose(k_embeds, [0, 2, 1, 3], name=f'k_permute_{self.name}')
    v_embeds = tf.transpose(v_embeds, [0, 2, 1, 3], name=f'v_permute_{self.name}')

    # alignment
    align = tf.matmul(q_embeds, k_embeds, transpose_b=True, name=f'align_matmul_{self.name}')
    align = tf.math.multiply(align, tf.math.sqrt(tf.cast(self.dk, tf.float32)), name=f'align_scale_{self.name}')
    align = tf.nn.softmax(align, axis=-1, name=f'align_softmax_{self.name}')

    # build context
    v_broadcast = tf.tile(v_embeds[:, :, tf.newaxis], [1, 1, self.seq_len, 1, 1], name=f'v_broadcast_{self.name}')
    v_aligned = tf.math.multiply(align[..., tf.newaxis], v_broadcast, name=f'v_aligned_{self.name}')
    summed_v = tf.reduce_sum(v_aligned, axis=3, name=f'summed_v_{self.name}')
    context = tf.transpose(summed_v, [0, 2, 3, 1], name=f'context_permute_{self.name}')
    context = tf.reshape(context, [-1, self.seq_len, self.num_heads * self.dv], name=f'context_reshape_{self.name}')

    # linear from concatenated heads
    x = self.linear(context)

    return x


  def get_config(self):
    config = super(MultiHeadSelfAttention, self).get_config()
    config.update(self.config)
    return config


class TransformerEncoderBlock(tf.keras.Model):
  """Multi-head self-attention and linear transformation, both
  with use of layer normalization and residual connections
  """
  def __init__(self, config, **kwargs):
    super(TransformerEncoderBlock, self).__init__(**kwargs)
    # config
    self.config = config
    self.attrs = ['dk', 'dv', 'seq_len', 'num_heads']
    for attr in self.attrs:
      assert attr in config, f"`{attr}` must be in config of `{self.__class__.__name__}` layer"
      setattr(self, attr, config[attr])
    self.mhsa = MultiHeadSelfAttention({
      'dk': self.dk,
      'dv': self.dv,
      'seq_len': self.seq_len,
      'num_heads': self.num_heads,
    })
    self.norm1 = tf.keras.layers.LayerNormalization(name=f'layer_norm1_{self.name}')
    self.boom = Boom({'dim': self.dv})
    self.norm2 = tf.keras.layers.LayerNormalization(name=f'layer_norm2_{self.name}')


  def call(self, inputs):
    """Inputs:
      x: [batch_size, seq_len, dv]
    """
    # TODO: activation functions
    x1 = inputs
    x2 = self.mhsa(x1)
    x2 = x1 + x2
    x2 = self.norm1(x2)
    # second block
    x3 = self.boom(x2)
    x3 = x3 + x2
    x3 = self.norm2(x3)
    return x3


  def get_config(self):
    config = super(MultiHeadSelfAttention, self).get_config()
    config.update(self.config)
    return config
