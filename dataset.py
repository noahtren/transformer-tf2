import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

from sp_tokenizer import get_sp_tokenizer


def write_as_txt():
  ds = tfds.load('wiki40b')['train']
  with open("wiki40b.txt", "a+") as f: 
    for val in tqdm(ds.take(150_000)):
      article = val['text'].numpy().decode()
      sentences = [s.strip() for s in article.split('. ')]
      f.write('\n'.join(sentences))
      # print(len(val['text'].numpy()))


tensor_specs = {
  'text': tf.string
}

features = tfds.features.FeaturesDict({
  'text': tf.string,
})



def serialize_example(inputs):
  """See https://www.tensorflow.org/tutorials/load_data/tfrecord
  """
  _serialized = {
    key: tf.io.serialize_tensor(val[key])
    for key in val.keys() if key in features
  }
  features.encode_example(_serialized)
  json.dumps(_serialized)


def parse_example(inputs):
  pass


def prep_dataset(context_length=1024):
  ds = tfds.load('wiki40b')['train']


def get_dataset():
  ds = tfds.load('wiki40b')['train']
  sp = get_sp_tokenizer()
  return ds, sp


if __name__ == "__main__":
  context_length = 128
  ds = tfds.load('wiki40b')['train']
  sp = get_sp_tokenizer()
  for val in ds.take(1):
    tokens = sp.encode(val['text'].numpy())
    tokens = tf.convert_to_tensor(tokens, dtype=tf.int16)
    tf.io.encode_proto(
      sizes=[0, 128],
      values=[tokens],
      field_names=['tokens']
    )
