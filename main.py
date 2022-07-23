import argparse
import os
import torch
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
from utils.load_data import load_dataset
from utils.logger import get_logger
from models.pho_beart import load_bpe, load_vocab, maping_word, make_masks, load_model
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm_notebook
from transformers.modeling_utils import *
from transformers import *
from sklearn.metrics import accuracy_score, f1_score

logger = get_logger()

parser = argparse.ArgumentParser("train.py")
parser.add_argument("--mode", help="available modes: train-test", required=True)
parser.add_argument("--train", help="train folder")
parser.add_argument("--test", help="test folder")
parser.add_argument("--s", help="path to save model")
args = parser.parse_args()

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    # return np.sum(pred_flat == labels_flat) / len(labels_flat)
    return accuracy_score(labels_flat, pred_flat), f1_score(labels_flat, pred_flat, average='micro')

def labling_data(y_transformer, y_train):
    y_train = [item for sublist in y_train for item in sublist]
    y_ids = y_transformer.fit_transform(y_train)
    return y_ids

if args.mode == "train-test":
    if not (args.train and args.test):
        parser.error("Mode train-test requires --train and --test")
    if not args.s:
        parser.error("Mode train-test requires --s ")
    train_path = os.path.abspath(args.train)
    test_path = os.path.abspath(args.test)

    print("Train model")
    model_path = os.path.abspath(args.s)
    print("Load data")
    X_train, y_train = load_dataset(train_path)
    X_test, y_test = load_dataset(test_path)
    target_names = list(set([i[0] for i in y_train]))
    y_transformer = LabelEncoder()
    y_train = labling_data(y_transformer, y_train)
    print("%d documents (training set)" % len(X_train))
    print("%d documents (test set)" % len(X_test))
    print("%d categories" % len(target_names))
    #split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, train_labels, val_labels = train_test_split(X_train, y_train, test_size=0.1)

    bpe = load_bpe()
    vocab = load_vocab()
    train_ids = maping_word(bpe, vocab, X_train, 256)
    val_ids = maping_word(bpe, vocab, X_val, 256)
    train_masks = make_masks(train_ids)
    val_masks = make_masks(val_ids)


    train_inputs = torch.tensor(train_ids)
    val_inputs = torch.tensor(val_ids)
    train_labels = torch.tensor(train_labels)
    val_labels = torch.tensor(val_labels)
    train_masks = torch.tensor(train_masks)
    val_masks = torch.tensor(val_masks)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)

    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=32)

    BERT_SA = load_model(len(target_names))
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    epochs = 10
    param_optimizer = list(BERT_SA.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, correct_bias=False)

    for epoch_i in range(0, epochs):
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        total_loss = 0
        BERT_SA.train()
        train_accuracy = 0
        nb_train_steps = 0
        train_f1 = 0

        for step, batch in tqdm_notebook(enumerate(train_dataloader)):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            BERT_SA.zero_grad()
            outputs = BERT_SA(b_input_ids,
                              token_type_ids=None,
                              attention_mask=b_input_mask,
                              labels=b_labels)
            loss = outputs[0]
            total_loss += loss.item()

            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            print(logits)
            print(label_ids)
            tmp_train_accuracy, tmp_train_f1 = flat_accuracy(logits, label_ids)
            train_accuracy += tmp_train_accuracy
            train_f1 += tmp_train_f1
            nb_train_steps += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(BERT_SA.parameters(), 1.0)
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print(" Accuracy: {0:.4f}".format(train_accuracy / nb_train_steps))
        print(" F1 score: {0:.4f}".format(train_f1 / nb_train_steps))
        print(" Average training loss: {0:.4f}".format(avg_train_loss))

        print("Running Validation...")
        BERT_SA.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        eval_f1 = 0
        for batch in tqdm_notebook(val_dataloader):
            batch = tuple(t.to(device) for t in batch)

            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = BERT_SA(b_input_ids,
                                  token_type_ids=None,
                                  attention_mask=b_input_mask)
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                tmp_eval_accuracy, tmp_eval_f1 = flat_accuracy(logits, label_ids)

                eval_accuracy += tmp_eval_accuracy
                eval_f1 += tmp_eval_f1
                nb_eval_steps += 1
        print(" Accuracy: {0:.4f}".format(eval_accuracy / nb_eval_steps))
        print(" F1 score: {0:.4f}".format(eval_f1 / nb_eval_steps))
    print("Training complete!")
