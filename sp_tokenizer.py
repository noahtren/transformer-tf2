import sentencepiece as spm


def train_tokenizer():
  spm.SentencePieceTrainer.Train(
    input='wiki40b.txt',
    model_prefix='sp',
    vocab_size=8_000,
  )


def get_sp_tokenizer(debug=False):
  sp = spm.SentencePieceProcessor()
  sp.load('sp.model')
  if debug:
    sp.encode("This is a test")
  return sp


if __name__ == "__main__":
  pass
