import tensorflow as tf
import tensorflow_text as text
from transformers import AutoTokenizer


def get_tf_tokenizer(hf_model_name, pad_length:int=128):
  """Get tokenization function which is compatible with TF graph mode.
  It's made as similar as possible to the HuggingFace BERT tokenizer (and others)
  by casting input strings to lowercase, normalizing them, and tokenizing with
  the same vocab.
  """
  # download huggingface tokenizer and get vocab
  hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
  vocab_list = list(hf_tokenizer.get_vocab().keys())
  vocab_list = [t if t.startswith("▁") else "##" + t for t in vocab_list]
  vocab_list = [t.replace("▁", "") for t in vocab_list]
  
  def create_table(vocab_list, num_oov=1):
    init = tf.lookup.KeyValueTensorInitializer(
        vocab_list,
        tf.range(tf.size(vocab_list, out_type=tf.int64), dtype=tf.int64),
        key_dtype=tf.string,
        value_dtype=tf.int64)
    return tf.lookup.StaticVocabularyTable(
        init, num_oov, lookup_key_dtype=tf.string)
  
  # make vocab accessible to TF.text tokenizer
  vocab_table = create_table(vocab_list)
  word_tokenizer = text.WhitespaceTokenizer()
  wp_tokenizer = text.WordpieceTokenizer(vocab_table, token_out_type=tf.int32)
  
  def tokenize_function(strings):
    strings = text.case_fold_utf8(strings)
    # strings = text.normalize_utf8(strings)
    words = word_tokenizer.tokenize(strings)
    tf_result = wp_tokenizer.tokenize(words)
    tf_result = tf_result.merge_dims(1, 2)
    bs = tf_result.shape[0]
    tf_result = tf_result.to_tensor(shape=[bs, pad_length])
    attn_mask = tf.where(tf_result == 0, tf.zeros_like(tf_result), tf.ones_like(tf_result))
    return {
      'input_ids': tf_result,
      'attention_mask': attn_mask
    }
  return hf_tokenizer, word_tokenizer, wp_tokenizer, tokenize_function

hf_tokenizer, word_tokenizer, wp_tokenizer, tokenize_function = get_tf_tokenizer("albert-base-v2", pad_length=16)

result = tokenize_function(["Hello, world!", "This is a test."])
graph_result = tf.function(tokenize_function)(["Hello, world!", "This is a test."])
hf_result = hf_tokenizer.batch_encode_plus(["Hello, world!", "This is a test."], add_special_tokens=False, padding='max_length', max_length=16, return_tensors='tf', return_token_type_ids=False)
