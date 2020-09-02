import code
import json

from transformers import AutoTokenizer
import tensorflow as tf
import tensorflow_text as text


def get_hf_tokenizer():
  model_name = "albert-base-v2"
  hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
  return hf_tokenizer


def get_tf_tokenizer():
  """Create TensorFlow tokenizer function which is compatible with graph mode.
  It is made as similar as possible to the original huggingface tokenizer.
  """
  hf_vocab = json.load(open("hf_vocab.json")).keys()
  hf_vocab = [s if s.startswith("▁") else "##" + s for s in hf_vocab]
  hf_vocab = [s.replace("▁", "") for s in hf_vocab]
  def create_table(vocab, num_oov=1):
    init = tf.lookup.KeyValueTensorInitializer(
        vocab,
        tf.range(tf.size(vocab, out_type=tf.int64), dtype=tf.int64),
        key_dtype=tf.string,
        value_dtype=tf.int64)
    return tf.lookup.StaticVocabularyTable(
        init, num_oov, lookup_key_dtype=tf.string)
  vocab_table = create_table(hf_vocab)
  pre_tokenizer = text.WhitespaceTokenizer()
  tf_tokenizer = text.WordpieceTokenizer(
    vocab_table,
    token_out_type=tf.int32)

  def tokenize_function(string):
    string = text.case_fold_utf8(string)
    string = text.normalize_utf8(string)
    w_result = basic_tokenizer.tokenize([string]).flat_values
    tf_result = tf_tokenizer.tokenize(w_result).flat_values
    return tf_result 

  return ws_tokenizer, wp_tokenizer, tokenize_function


def write_hf_vocab():
  with open("hf_vocab.json", "w+") as f:
    f.write(json.dumps(hf_tokenizer.get_vocab()))


def test_tokenizers(test_string, tf_tokenizer, hf_tokenizer):
  tf_result = tf_tokenizer(test_string).numpy().tolist()
  hf_result = hf_tokenizer.encode(test_string, add_special_tokens=False)
  code.interact(local={**locals(), **globals()})
  tf.reduce_any(tf_result == hf_result)

if __name__ == "__main__":
  hf_tokenizer = get_hf_tokenizer()
  # write_hf_vocab()
  _, _, tf_tokenizer = get_tf_tokenizer()
  test_tokenizers("wallaby", tf_tokenizer, hf_tokenizer)
