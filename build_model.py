# /**
#  * @author Zhirui(Alex) Yang
#  * @email zy2494@columbia.edu
#  * @create date 2022-12-08 12:55:32
#  * @modify date 2022-12-08 12:55:32
#  * @desc [This file includes some common functions for processing data and build model]
#  */


from utils import TAG2IDX, IDX2TAG, WSJ_TAGS

import os
from collections import Counter
import pandas as pd
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm_notebook as tqdm

import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer, BertModel
from torchmetrics.functional.classification import multiclass_f1_score, multiclass_precision, multiclass_recall, multiclass_accuracy

torch.manual_seed(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOKENIZER = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)


class PosDataset(data.Dataset):
    def __init__(self, word_lst, tag_lst):
        sents, tags_li = [], [] # list of lists
        for i in range(len(word_lst)):
            sents.append(["[CLS]"] + word_lst[i] + ["[SEP]"])
            tags_li.append(["<pad>"] + tag_lst[i] + ["<pad>"])
        self.sents, self.tags_li = sents, tags_li

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx] # words, tags: string list

        # We give credits only to the first piece.
        x, y = [], [] # list of ids
        is_heads = [] # list. 1: the token is the first piece of a word
        for w, t in zip(words, tags):
            tokens = TOKENIZER.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = TOKENIZER.convert_tokens_to_ids(tokens)

            is_head = [1] + [0]*(len(tokens) - 1)

            t = [t] + ["<pad>"] * (len(tokens) - 1)  # <PAD>: no decision
            yy = [TAG2IDX[each] for each in t]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)

        assert len(x)==len(y)==len(is_heads), "len(x)={}, len(y)={}, len(is_heads)={}".format(len(x), len(y), len(is_heads))

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen


def pad(batch):
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f(1, maxlen)
    y = f(-2, maxlen)


    f = torch.LongTensor

    return words, f(x), is_heads, tags, f(y), seqlens


class Net(nn.Module):
    def __init__(self, vocab_size=None):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')

        self.fc = nn.Linear(768, vocab_size)
        self.device = DEVICE

    def forward(self, x, y):
        '''
        x: (N, T). int64
        y: (N, T). int64
        '''
        x = x.to(self.device)
        y = y.to(self.device)
        
        if self.training:
            self.bert.train()
            encoded_layers, _ = self.bert(x)
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(x)
                enc = encoded_layers[-1]
        
        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat


def train(model, iterator, optimizer, criterion):
    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens = batch
        _y = y # for monitoring
        optimizer.zero_grad()
        logits, y, _ = model(x, y) # logits: (N, T, VOCAB), y: (N, T)

        logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)

        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()

        if i%10==0: # monitoring
            print("step: {}, loss: {}".format(i, loss.item()))


def eval(model, iterator, save_output=True, output_file=None, average="weighted", num_classes=len(WSJ_TAGS), idx2tag=IDX2TAG):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []

    pred_lst = []
    true_lst = []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch

            _, _, y_hat = model(x, y)  # y_hat: (N, T)

            for s in y_hat.cpu().numpy().tolist():
              pred_lst.extend(s)
            for s in y.numpy().tolist():
              true_lst.extend(s)

            if save_output:
              Words.extend(words)
              Is_heads.extend(is_heads)
              Tags.extend(tags)
              Y.extend(y.numpy().tolist())
              Y_hat.extend(y_hat.cpu().numpy().tolist())

    ## gets results and save
    if save_output:
      with open(output_file, 'w') as fout:
          for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
              y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
              preds = [idx2tag[hat] for hat in y_hat]
              assert len(preds)==len(words.split())==len(tags.split())
              for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                  fout.write("{} {} {}\n".format(w, t, p))
              fout.write("\n")
            
    # ## calc metric
    # y_true =  np.array([tag2idx[line.split()[1]] for line in open('result', 'r').read().splitlines() if len(line) > 0])
    # y_pred =  np.array([tag2idx[line.split()[2]] for line in open('result', 'r').read().splitlines() if len(line) > 0])


    precision_value = multiclass_precision(
            torch.tensor(pred_lst), torch.tensor(true_lst), num_classes=num_classes, ignore_index=0, 
            average=average)   
    recall_value = multiclass_recall(
            torch.tensor(pred_lst), torch.tensor(true_lst), num_classes=num_classes, ignore_index=0, 
            average=average)   
    f1_value = multiclass_f1_score(
            torch.tensor(pred_lst), torch.tensor(true_lst), num_classes=num_classes, ignore_index=0, 
            average=average)   
    acc = multiclass_accuracy(
        torch.tensor(pred_lst), torch.tensor(true_lst), num_classes=num_classes, ignore_index=0, 
        average=average)    

    precision_value_micro = multiclass_precision(
            torch.tensor(pred_lst), torch.tensor(true_lst), num_classes=num_classes, ignore_index=0, 
            average="micro")   
    recall_value_micro = multiclass_recall(
            torch.tensor(pred_lst), torch.tensor(true_lst), num_classes=num_classes, ignore_index=0, 
            average="micro")   
    f1_value_micro = multiclass_f1_score(
            torch.tensor(pred_lst), torch.tensor(true_lst), num_classes=num_classes, ignore_index=0, 
            average="micro")   
    acc_micro = multiclass_accuracy(
        torch.tensor(pred_lst), torch.tensor(true_lst), num_classes=num_classes, ignore_index=0, 
        average="micro")    


    precision_value_macro = multiclass_precision(
            torch.tensor(pred_lst), torch.tensor(true_lst), num_classes=num_classes, ignore_index=0, 
            average="macro")   
    recall_value_macro = multiclass_recall(
            torch.tensor(pred_lst), torch.tensor(true_lst), num_classes=num_classes, ignore_index=0, 
            average="macro")   
    f1_value_macro = multiclass_f1_score(
            torch.tensor(pred_lst), torch.tensor(true_lst), num_classes=num_classes, ignore_index=0, 
            average="macro")   
    acc_macro = multiclass_accuracy(
        torch.tensor(pred_lst), torch.tensor(true_lst), num_classes=num_classes, ignore_index=0, 
        average="macro")    



    return precision_value, recall_value, f1_value, acc, precision_value_micro, recall_value_micro, f1_value_micro, acc_micro, precision_value_macro, recall_value_macro, f1_value_macro, acc_macro


