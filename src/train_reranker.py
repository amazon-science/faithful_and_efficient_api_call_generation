#!/usr/bin/env python
# coding: utf-8

import os

import argparse
import sys
import pickle as pk

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer

import re
import numpy as np
import random
import math
import json

from nltk.tokenize import word_tokenize

from datasets import Dataset
from datasets import load_metric
from torch.utils.data import DataLoader

from torch.optim import AdamW
from transformers import get_scheduler

from tqdm.auto import tqdm

from packages.fast_soft_sort.pytorch_ops import soft_rank

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", default=None, type=str,
                    help="Optional use of config file for passing the arguments")
parser.add_argument("--output_dir", default=None, type=str,
                    help="The output directory where the model predictions and checkpoints will be written")
parser.add_argument("--gpu_device", default=0, type=int,
                    help="GPU device id")
parser.add_argument("--train_set", default=None, type=str,
                    help="Path to the file of training data")
parser.add_argument("--test_set", default=None, type=str,
                    help="Path to the file of testing data")
parser.add_argument("--max_input_length", default=512, type=int,
                    help="Maximum input length after tokenization")
parser.add_argument("--do_train", action='store_true',
                    help="Whether to run training.")
parser.add_argument("--seed", default=1, type=int,
                    help="Random seed")
parser.add_argument("--num_epochs", default=5, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                    help="Batch size per GPU/CPU for training")
parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int,
                    help="Batch size per GPU/CPU for evaluation")
parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                    help="Number of updates steps to accumulate before performing a backward/update pass; effective training batch size equals to per_gpu_train_batch_size * gradient_accumulation_steps")
parser.add_argument("--learning_rate", default=5e-5, type=float,
                    help="The initial learning rate of AdamW for the model")
parser.add_argument('--scheduler_type', type=str, default="linear",
                    help="Learning rate scheduler type (linear/cosine/cosine_with_restarts/polynomial/constant/constant_with_warmup)")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Scheduler warmup steps")

args = parser.parse_args()

if args.config_file != None:
    with open(args.config_file, 'r') as f:
        args.__dict__ = json.load(f)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

print(args.__dict__)
with open(os.path.join(args.output_dir,'config.dict'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
device = torch.device('cuda:'+str(args.gpu_device)) if torch.cuda.is_available() else torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
tokenizer.truncation_side='left'

model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=1)
model.to(device)

train = pk.load(open(args.train_set,'rb'))
test = pk.load(open(args.train_set,'rb'))

train_set = Dataset.from_dict({'text':[item[0] for item in train],'labels':[item[1] for item in train]})
test_set = Dataset.from_dict({'text':[item[0] for item in test],'labels':[item[1] for item in test]})

def tokenize_content(sample):
    return tokenizer(sample['text'], padding = 'max_length', truncation=True, max_length=args.max_input_length, return_tensors='pt')

def process_dataset(data):
    agg_test = {}
    for sample in tqdm(data,total = len(data)):
        key=sample['text'].split('\nAPI:')[0]
        tc = tokenize_content(sample)
        tc['labels']=torch.tensor([sample['labels']],dtype=torch.float32)
        if key in agg_test:
            agg_test[key].append(tc)
        else:
            agg_test[key]=[tc]
    return agg_test

train_set = process_dataset(train_set)
test_set = process_dataset(test_set)

def prepare_batch(data):
    batchified_data = []
    drop_count=0
    for d in tqdm(list(data.values())):
        if len(d)>1:
            bcd = {}
            for k in d[0].keys():
                temp = [ele[k] for ele in d]
                bcd[k] = torch.cat(temp)
            if sum(bcd['labels'])>0:
                batchified_data.append(bcd)
            else:
                drop_count+=1
        else:
            drop_count+=1
    print(drop_count)
    return batchified_data

batchified_train = prepare_batch(train_set)
batchified_test = prepare_batch(test_set)

def spearman_ranking_loss(y_pred, y_true):
    n = y_true.size()[0]
    soft_rankings = soft_rank(torch.stack([y_pred, y_true]).cpu(), regularization_strength=2.0).to(device)
    rank_pred = soft_rankings[0]
    rank_true = soft_rankings[1]
    d = rank_pred - rank_true
    rho = 1-(6*torch.sum(d**2))/(n*(n**2-1)) 
    return 1-rho

optimizer = AdamW(list(model.parameters()),lr=args.learning_rate)

num_training_steps = args.num_epochs * len(batchified_train)
    
lr_scheduler = get_scheduler(
    args.scheduler_type,
    optimizer = optimizer,
    num_warmup_steps = args.warmup_steps,
    num_training_steps = num_training_steps
)

epoch_mse_loss = [[]]*args.num_epochs
epoch_cor_loss = [[]]*args.num_epochs

epoch_test_mse_loss = [[]]*args.num_epochs
epoch_test_cor_loss = [[]]*args.num_epochs

for epoch in tqdm(range(0,args.num_epochs)):
    
    step=0
    
    progress_bar = tqdm(range(len(batchified_train)))
    
    effective_batch_mse = []
    effective_batch_cor = []
    
    random.shuffle(batchified_train)
    for batch in batchified_train:

        step+=1

        model_input = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**model_input)
        
        cor_loss = spearman_ranking_loss(outputs.logits.view(-1),model_input['labels'])*100
        
        effective_batch_mse.append(outputs.loss.item())
        effective_batch_cor.append(cor_loss.item())
        
        total_loss = outputs.loss/args.gradient_accumulation_steps + cor_loss/args.gradient_accumulation_steps
                        
        total_loss.backward()
        
        if args.gradient_accumulation_steps and not step % args.gradient_accumulation_steps:
            progress_bar.set_description("MSE Loss: %f COR Loss: %f" %(np.mean(effective_batch_mse),
                                                                       np.mean(effective_batch_cor)))
                
            progress_bar.update(args.gradient_accumulation_steps)
            
            epoch_mse_loss[epoch]+=effective_batch_mse
            epoch_cor_loss[epoch]+=effective_batch_cor
            
            effective_batch_mse = []
            effective_batch_cor = []
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
    print('\n\nEpoch-%d Train MSE: %f'%(epoch,np.mean(epoch_mse_loss[epoch])))
    print('Epoch-%d Train COR: %f'%(epoch,np.mean(epoch_cor_loss[epoch])))
                                
    for batch in tqdm(batchified_test):

        model_input = {k: v.to(device) for k, v in batch.items()}
                                         
        with torch.no_grad(): 
            outputs = model(**model_input)

        cor_loss = spearman_ranking_loss(outputs.logits.view(-1),model_input['labels'])
        
        epoch_test_mse_loss[epoch].append(outputs.loss.item())
        epoch_test_cor_loss[epoch].append(cor_loss.item())
                                
    print('\n\nEpoch-%d Test MSE: %f'%(epoch,np.mean(epoch_test_mse_loss[epoch])))
    print('Epoch-%d Test COR: %f'%(epoch,np.mean(epoch_test_cor_loss[epoch])))
    
    epoch_save_path = os.path.join(args.output_dir,str(epoch))
    if not os.path.exists(epoch_save_path):
        os.makedirs(epoch_save_path)
    model.save_pretrained(epoch_save_path)

tokenizer.save_pretrained(args.output_dir)
model.save_pretrained(args.output_dir)
