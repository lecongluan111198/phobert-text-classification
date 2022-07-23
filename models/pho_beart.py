from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import RobertaForSequenceClassification, RobertaConfig, AdamW, BertForSequenceClassification
import argparse

def load_bpe():
  parser = argparse.ArgumentParser()
  parser.add_argument('--bpe-codes',
      default="../PhoBERT_base_transformers/bpe.codes",
      required=False,
      type=str,
      # help='path to fastBPE BPE'
  )
  args, unknown = parser.parse_known_args()
  bpe = fastBPE(args)
  return bpe

def load_vocab():
  vocab = Dictionary()
  vocab.add_from_file("../PhoBERT_base_transformers/dict.txt")
  return vocab

def load_model(num_labels):
  config = RobertaConfig.from_pretrained(
    "../PhoBERT_base_transformers/config.json", from_tf=False, num_labels=num_labels,
    output_hidden_states=False,
  )
  BERT_SA = BertForSequenceClassification.from_pretrained(
    "../PhoBERT_base_transformers/model.bin",
    config=config
  )
  # BERT_SA.cuda()
  return BERT_SA

def maping_word(bpe, vocab, data, maxlen = 125):
  train_ids = []
  for sent in data:
    subwords = '<s> ' + bpe.encode(sent) + ' </s>'
    encoded_sent = vocab.encode_line(subwords, append_eos=True, add_if_not_exist=False).long().tolist()
    train_ids.append(encoded_sent)
  train_ids = pad_sequences(train_ids, maxlen=maxlen, dtype="long", value=0, truncating="post", padding="post")
  return train_ids

def make_masks(ids):
  masks = []
  for sent in ids:
    mask = [int(token_id > 0) for token_id in sent]
    masks.append(mask)
  return masks

