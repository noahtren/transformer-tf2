import code

import tensorflow as tf

from attn import TransformerEncoderBlock
from dataset import get_dataset


def positional_encoding(seq_len, d_v):
  """Returns positional encodings with shape
  [seq_len, d_v], where each position in the sequence has a
  unique positional encoding.

  Based on this article: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
  """
  ks = tf.range(d_v // 2, dtype=tf.float32)
  # dimensions for a single position
  pos_dims = 1 / (10_000) ** (2 * ks / d_v)
  # map to multiple dimensions
  position_terms = tf.range(seq_len, dtype=tf.float32)[tf.newaxis] * pos_dims[:, tf.newaxis]
  sins = tf.math.sin(position_terms)
  coss = tf.math.cos(position_terms)
  pos = tf.stack([sins, coss], axis=2)
  pos = tf.reshape(pos, [seq_len, -1])
  return pos


class Transformer(tf.keras.Model):
  def __init__(self, config):
    super(Transformer, self).__init__()
    self.attrs = ['num_blocks', 'vocab_size', 'dk', 'dv', 'seq_len', 'num_heads']
    for attr in self.attrs:
      assert attr in config, f"`{attr}` must be in config of `{self.__class__.__name__}` layer"
      setattr(self, attr, config[attr])
    self.token_embed = tf.keras.layers.Dense(self.dv)
    self.blocks = [TransformerEncoderBlock({
      'dk': self.dk,
      'dv': self.dv,
      'seq_len': self.seq_len,
      'num_heads': self.num_heads,
    }) for _ in range(self.num_blocks)]

  def call(self, inputs):
    tokens = inputs
    tokens = tf.one_hot(tokens, depth=self.vocab_size, dtype=tf.float32)
    x = self.token_embed(tokens)
    pos = positional_encoding(self.seq_len, self.dv)
    code.interact(local={**locals(), **globals()})
    x = x + pos
    for block in self.blocks:
      x = block(x)
    return x


def trace_and_visualize(func, data, vis_name, logdir='./logs'):
  tf.summary.trace_on()
  writer = tf.summary.create_file_writer(logdir)
  # make graph mode function, and run
  tf_func = tf.function(func)
  tf_func(data)
  with writer.as_default():
    tf.summary.trace_export(name=vis_name, step=0)


if __name__ == "__main__":
  ds, sp = get_dataset()
  for val in ds.take(1):
    text = sp.encode(val['text'].numpy())
    text = text[:128]
    text = tf.convert_to_tensor(text)[tf.newaxis]
    text = tf.tile(text, [2, 1])

  transformer = Transformer({
    'num_blocks': 2,
    'dk': 32,
    'dv': 64,
    'seq_len': 128,
    'num_heads': 4,
    'vocab_size': sp.vocab_size()
  })
  # trace_and_visualize(transformer, text, vis_name='transformer')
  transformer(text)
