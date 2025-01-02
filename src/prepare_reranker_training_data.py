#!/usr/bin/env python
# coding: utf-8

import pickle as pk
import datasets
from collections import Counter
import re
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(666)

def fn(**kwargs):
    return kwargs

def get_api_call(model_output):
    api_call_pattern = r"\[(\w+)\((.*)\)\]"
    api_call_pattern = re.compile(api_call_pattern)
    match = api_call_pattern.search(model_output)
    if match:
        return match.group(0)
    else:
        return None

def parse_api_call(text):
    pattern = r"\[(\w+)\((.*)\)\]"
    match = re.search(pattern, text, re.MULTILINE)

    api_name = match.group(1)
    params = match.group(2)

    param_pattern = r"(\w+)\s*=\s*['\"](.+?)['\"]|(\w+)\s*=\s*(\[.*\])|(\w+)\s*=\s*(\w+)"
    param_dict = {}
    for m in re.finditer(param_pattern, params):
        if m.group(1):
            param_dict[m.group(1)] = m.group(2)
        elif m.group(3):
            param_dict[m.group(3)] = m.group(4)
        elif m.group(5):
            param_dict[m.group(5)] = m.group(6)
    return api_name, param_dict

def prepare_elements(call):
    parsed = parse_api_call(call)
    name, parameter_values = parsed
    elements = [name]
    for k, v in parameter_values.items():
        elements.append(k+'='+v)
    return elements

def code_element_score(list1, list2):
    len1=len(list1)
    len2=len(list2)
    counter1 = Counter(list1)
    counter2 = Counter(list2)
    unique_to_list1 = []
    for elem, count in counter1.items():
        unique_count = count - counter2.get(elem, 0)
        if unique_count > 0:
            unique_to_list1.extend([elem] * unique_count)
    unique_to_list2 = []
    for elem, count in counter2.items():
        unique_count = count - counter1.get(elem, 0)
        if unique_count > 0:
            unique_to_list2.extend([elem] * unique_count)

    final_score = 1 - (len(unique_to_list1)+len(unique_to_list2))*1.0/(len1+len2)
    return final_score


inferences = pk.load(open(input('Path to the Pickle File of Beam Searched Trainset Candidates: '),'rb'))

random.shuffle(inferences)
trainset = []
flag=False
flag_count = 0
for sample in tqdm(inferences[:int(len(inferences)*0.8)]):
    scorer_context = sample[0][0].split('\nAPI descriptions:')[-1].split('Generate API Request:')[0].strip()
    gold = sample[0][1]
    gold_elements = prepare_elements(gold)
    for candidate in sample[1]:        
        try:
            cand_elements = prepare_elements(candidate)
            sc = code_element_score(gold_elements,cand_elements)
        except:
            cand_elements = []
            sc = 0
        if not sc:
            flag_count+=1
        trainset.append((scorer_context + '\nAPI:' + candidate,sc))
print(flag_count)
pk.dump(trainset,open('api_bank-scorer-train.pk','wb'))

testset = []
flag=False
flag_count = 0
for sample in tqdm(inferences[int(len(inferences)*0.8):]):
    scorer_context = sample[0][0].split('\nAPI descriptions:')[-1].split('Generate API Request:')[0].strip()
    gold = sample[0][1]
    gold_elements = prepare_elements(gold)
    for candidate in sample[1]:        
        try:
            cand_elements = prepare_elements(candidate)
            sc = code_element_score(gold_elements,cand_elements)
        except:
            cand_elements = []
            sc = 0
        if not sc:
            flag_count+=1
        testset.append((scorer_context + '\nAPI:' + candidate,sc))
print(flag_count)
pk.dump(testset,open('api_bank-scorer-test.pk','wb'))
