#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import sys
import pickle as pk

from tqdm import tqdm

import json

import re
from nltk import sent_tokenize

from datasets import Dataset
from transformers import AutoTokenizer

from datetime import datetime, timedelta

parser = argparse.ArgumentParser()

parser.add_argument("--config_file", default=None, type=str,
                    help="Optional use of config file for passing the arguments")
parser.add_argument("--output_dir", default=None, type=str,
                    help="The output directory where the processed constraints will be written")
parser.add_argument("--model", default=None, type=str,
                    help="Huggingface model name / Recorded model name / local path to model directory")
parser.add_argument("--api_data", default=None, type=str,
                    help="Recorded api data name / Path to API data")

args = parser.parse_args()

if args.config_file != None:
    with open(args.config_file, 'r') as f:
        args.__dict__ = json.load(f)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

print(args.__dict__)
with open(os.path.join(args.output_dir,'config.dict'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)


model_records = {
                 'alpaca-13b-gpt4':'chavinlo/gpt4-x-alpaca',
                 'lynx-7b':'liminghao1630/Lynx-7b',
                 }

model_path = args.model_path
if model_path in model_records:
    model_path = model_records[model_path]

dataset_records = {
                   'api_bank':'$path to api bank test-data/level-1-api.json available at https://huggingface.co/datasets/liminghao1630/API-Bank$', 
                   }

data_path = args.dataset_path
if data_path in dataset_records:
    data_path = dataset_records[data_path]

tokenizer = AutoTokenizer.from_pretrained(model_path)

def getTokenIds(target):
    start_idx = 1
    if args.use_alexa_model:
        start_idx=2
    target_tokens = tokenizer.tokenize('="'+target+' "')[start_idx:-1]
    return [tokenizer.convert_tokens_to_ids(tk) for tk in target_tokens]
    
def getTokenIdsWithLeadingSpace(target):
    start_idx = 1
    if args.use_alexa_model:
        start_idx=2
    target_tokens = tokenizer.tokenize('=" '+target+' "')[start_idx:-1]
    return [tokenizer.convert_tokens_to_ids(tk) for tk in target_tokens]

def construct_tree(lists):
    root = {}
    for li in tqdm(lists):
        node = root
        for token_id in li:
            if token_id not in node:
                node[token_id]={}
            node = node[token_id]
    return root

if 'api_bank' in args.api_data:
    
    PREFIX = tokenizer.encode("API-Request: [")[1:]
    SUFFIX = tokenizer.convert_tokens_to_ids(']')
    LEFT_BRACKET = tokenizer.convert_tokens_to_ids('(')
    RIGHT_BRACKET = tokenizer.convert_tokens_to_ids(')')
    
    OPEN_PARA = tokenizer.convert_tokens_to_ids("='")
    CLOSE_PARA_COMMA = tokenizer.convert_tokens_to_ids("',")
    CLOSE_PARA_RIGHT_BRACKET = tokenizer.convert_tokens_to_ids("')")
        
    OPEN_STR_LIST = tokenizer.convert_tokens_to_ids("['")
    CLOSE_STR_LIST = tokenizer.convert_tokens_to_ids("']")
    
    OPEN_DICT_LIST = tokenizer.convert_tokens_to_ids("[")
    OPEN_DICT_LIST2 = tokenizer.convert_tokens_to_ids("{")
    CLOSE_DICT_LIST = tokenizer.convert_tokens_to_ids("}]")
    
else: # dstc8
    
    DOT = tokenizer.convert_tokens_to_ids('.')
    LEFT_BRACKET = tokenizer.convert_tokens_to_ids('(')
    RIGHT_BRACKET = tokenizer.convert_tokens_to_ids(')')
    EQUAL_OPEN_QUOTE = tokenizer.convert_tokens_to_ids('="')
    CLOSE_QUOTE_COMMA = tokenizer.convert_tokens_to_ids('",')
    CLOSE_QUOTE_RIGHT_BRACKET = tokenizer.convert_tokens_to_ids('")')
    CLOSE_QUOTE = tokenizer.convert_tokens_to_ids('"')
    COMMA = tokenizer.convert_tokens_to_ids(',')

if data_path.endswith('.arrow'):
    loaded_api_data = Dataset.from_file(data_path)
else:
    try:
        loaded_api_data = json.load(open(data_path,'r'))
    except:
        try:
            loaded_api_data = json.loads(open(data_path,'r').readline())
        except:
            loaded_api_data = json.loads(' '.join(open(data_path,'r').readlines()))

if 'bank' in args.api_data:

    ## api bank date time formatting constraints
    def enumerate_dates(start_date, end_date):
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        date_list = []

        delta = end - start

        for i in range(delta.days + 1):
            day = start + timedelta(days=i)
            date_list.append(day.strftime("%Y-%m-%d"))

        return date_list

    start_date = "2023-01-01"
    end_date = "2023-12-31"
    possible_yyyymmdd = enumerate_dates(start_date, end_date)


    def enumerate_times(start_time, end_time, interval_seconds=1):
        start = datetime.strptime(start_time, "%H:%M:%S")
        end = datetime.strptime(end_time, "%H:%M:%S")
        time_list = []

        while start <= end:
            time_list.append(start.strftime("%H:%M:%S"))
            start += timedelta(seconds=interval_seconds)

        return time_list

    start_time = "00:00:00"
    end_time = "23:59:59"
    possible_hhmmss = enumerate_times(start_time, end_time, interval_seconds=1)


    def enumerate_month_dates(start_date, end_date):
        year = "2000"
        start = datetime.strptime(year + "-" + start_date, "%Y-%m-%d")
        end = datetime.strptime(year + "-" + end_date, "%Y-%m-%d")
        date_list = []

        delta = end - start

        for i in range(delta.days + 1):
            day = start + timedelta(days=i)
            date_list.append(day.strftime("%m-%d"))

        return date_list

    start_date = "01-01"
    end_date = "12-31"
    possible_mmdd = enumerate_month_dates(start_date, end_date)

    d_plus_t = []
    for d in tqdm(possible_yyyymmdd):
        for t in possible_hhmmss:
            d_plus_t.append(d+' '+t)

    possible_hhmmss_ids = []
    for pv in tqdm(possible_hhmmss):
        temp = getTokenIds(pv)
        possible_hhmmss_ids.append(temp+[CLOSE_PARA_COMMA])
        possible_hhmmss_ids.append(temp+[CLOSE_PARA_RIGHT_BRACKET])

    possible_hhmmss_tree = construct_tree(possible_hhmmss_ids)

    possible_yyyymmdd_ids = []
    for pv in tqdm(possible_yyyymmdd):
        temp = getTokenIds(pv)
        possible_yyyymmdd_ids.append(temp+[CLOSE_PARA_COMMA])
        possible_yyyymmdd_ids.append(temp+[CLOSE_PARA_RIGHT_BRACKET])
    possible_yyyymmdd_tree = construct_tree(possible_yyyymmdd_ids)

    possible_mmdd_ids = []
    for pv in tqdm(possible_mmdd):
        temp = getTokenIds(pv)
        possible_mmdd_ids.append(temp+[CLOSE_PARA_COMMA])
        possible_mmdd_ids.append(temp+[CLOSE_PARA_RIGHT_BRACKET])

    possible_mmdd_tree = construct_tree(possible_mmdd_ids)

    possible_d_plus_t_ids = []
    for pv in tqdm(d_plus_t):
        temp = getTokenIds(pv)
        possible_d_plus_t_ids.append(temp+[CLOSE_PARA_COMMA])
        possible_d_plus_t_ids.append(temp+[CLOSE_PARA_RIGHT_BRACKET])

    possible_d_plus_t_tree = construct_tree(possible_d_plus_t_ids)

    possible_space_hhmmss_ids = []
    for pv in tqdm(possible_hhmmss):
        temp = getTokenIdsWithLeadingSpace(pv)
        possible_space_hhmmss_ids.append(temp+[CLOSE_PARA_COMMA])
        possible_space_hhmmss_ids.append(temp+[CLOSE_PARA_RIGHT_BRACKET])
    possible_space_hhmmss_tree = construct_tree(possible_space_hhmmss_ids)
    
    ## API_BANK
    apis = []
    for sample in tqdm(loaded_api_data,total=len(loaded_api_data)):
        if 'train' in args.api_data:
            for line in sample['input'].split('\n'):
                if len(line)>3 and line[0]=='{' and line not in apis:
                    apis.append(line)
        else:
            for line in sample['instruction'].split('API descriptions:')[-1].split('\n'):
                if len(line)>3 and line not in apis:
                    apis.append(line)

    api_doc = {}
    for api_def in tqdm(apis):
        api_def = json.loads(api_def)
        
        if 'train' in args.api_data:
            func = api_def['apiCode']
        else:
            func = api_def['name']

        api_doc[func]={}
        api_doc[func]['para_pv_ids']={}
        api_doc[func]['required_para']=[]
        api_doc[func]['all_para']=[]

        para_open_regular = []
        para_open_list_str = []
        para_open_list_dict = []

        para_keyword = 'input_parameters'
        if 'train' in args.api_data:
            para_keyword = 'parameters'
            
        for para in api_def[para_keyword]:

            api_doc[func]['required_para'].append(para) 
            api_doc[func]['all_para'].append(para)

            para_type = api_def[para_keyword][para]['type']

            if para_type=='list(str)':
                para_open_list_str.append(para)
            elif para_type=='list':
                para_open_list_dict.append(para)
            else:
                para_open_regular.append(para)
                if para_type == 'bool':
                    pv_list = ['True','False']
                    cqc = [getTokenIds(pv)+[CLOSE_PARA_COMMA] for pv in pv_list]
                    cqrb = [getTokenIds(pv)+[CLOSE_PARA_RIGHT_BRACKET] for pv in pv_list]
                    api_doc[func]['para_pv_ids'][para]=construct_tree(cqc+cqrb)

            if '_count' in para:
                pv_list = ['0','1','2','3','4','5','6','7','8','9']
                cqc = [getTokenIds(pv)+[CLOSE_PARA_COMMA] for pv in pv_list]
                cqrb = [getTokenIds(pv)+[CLOSE_PARA_RIGHT_BRACKET] for pv in pv_list]
                api_doc[func]['para_pv_ids'][para]=construct_tree(cqc+cqrb)
            
            elif 'Format: %m-%d' in api_def[para_keyword][para]['description']:
                api_doc[func]['para_pv_ids'][para]=possible_mmdd_tree
            elif 'Format: %Y-%m-%d"' in api_def[para_keyword][para]['description']:
                api_doc[func]['para_pv_ids'][para]=possible_yyyymmdd_tree
            elif 'Format: %Y-%m-%d %H:%M:%S' in api_def[para_keyword][para]['description']:
                api_doc[func]['para_pv_ids'][para]=possible_d_plus_t_tree
                
                
            para_open_regular_ids = [getTokenIds(para)+[OPEN_PARA] for para in para_open_regular]
            para_open_list_str_ids = [getTokenIds(para)+[OPEN_PARA, OPEN_STR_LIST] for para in para_open_list_str]
            para_open_list_dict_ids = [getTokenIds(para)+[OPEN_PARA, OPEN_DICT_LIST, OPEN_DICT_LIST2] for para in para_open_list_dict]

            api_doc[func]['first_para_gen_options'] = construct_tree(para_open_regular_ids
                                                                     +para_open_list_str_ids
                                                                     +para_open_list_dict_ids
                                                                     +[[RIGHT_BRACKET]]
                                                                     )

            para_open_regular_ids = [getTokenIdsWithLeadingSpace(para)+[OPEN_PARA] for para in para_open_regular]
            para_open_list_str_ids = [getTokenIdsWithLeadingSpace(para)+[OPEN_PARA, OPEN_STR_LIST] for para in para_open_list_str]
            para_open_list_dict_ids = [getTokenIdsWithLeadingSpace(para)+[OPEN_PARA, OPEN_DICT_LIST, OPEN_DICT_LIST2] for para in para_open_list_dict]

            api_doc[func]['para_gen_options'] = construct_tree(para_open_regular_ids
                                                                     +para_open_list_str_ids
                                                                     +para_open_list_dict_ids
                                                                     )

    pk.dump(api_doc, open(os.path.join(args.output_dir,'api_constraints-tree.pk'),'wb'))    

    lists = [getTokenIds(func)+[LEFT_BRACKET] for func in list(api_doc.keys())]
    func_pv = construct_tree(lists)

    pk.dump(func_pv, open(os.path.join(args.output_dir,'api_constraints-func_pv-tree.pk'),'wb'))

else:
    ## DSTC8
    apis = []
    for i in tqdm(loaded_api_data,total=len(loaded_api_data)):
        prompt = '\n'.join([ele.strip() for ele in i['turns'][:-1]])
        api_stacks = prompt.split("You have access to the user's context and the following APIs.")[1].split("Example workflow similar to current utterance:")[0]
        api_stacks = api_stacks.split('\n')
        for line in api_stacks:
            if len(line)>10:
                line = line.strip()
                if line not in apis:
                    apis.append(line)

    
    records = []
    out = open(os.path.join(args.output_dir,'constraints.txt'),'w')
    for api in apis:
        try:
            matches = re.match(r'([^\.]+)\.([a-zA-z]+)\(([^\.)]+)\)',api.split('. . ')[0])
            out.write('>>> ' + matches.group(1))
            out.write('\n')
            out.write('[S] ' + matches.group(1) + ' [R] has function [O] ' + matches.group(2))
            out.write('\n')
            out.write('>>>>>> ' + matches.group(2))
            out.write('\n')
            for para in matches.group(3).split(', '):
                elements = para.split(':')
                out.write('[S] ' + matches.group(2) + ' [R] '+ elements[1].strip().strip("\"").lower() + ' parameter [O] '+ elements[0].strip().strip("\""))
                out.write('\n')
            for api_sent in api.split('. . ')[1:]:
                for sent in sent_tokenize(api_sent.replace('\",','[PLACEHOLDER]').replace(', ','. ')):
                    if len(sent)>0:
                        para_type = re.match(r'\"([^\"]+)\" is a ([^\s]+) type\.', sent)
                        if para_type:
                            out.write('[S] ' + para_type.group(1) + ' [R] data type [O] '+ para_type.group(2))
                            out.write('\n')
                        else:
                            definition = re.match(r'\"([^\"]+)\" refers to ([^\.]+)\.', sent)
                            if definition:
                                out.write('[S] ' + definition.group(1) + ' [R] definition [O] '+ definition.group(2))
                                out.write('\n')
                            else:
                                value_type = re.match(r'\"([^\"]+)\" is a ([^\s]+) value\.', sent)
                                if value_type:
                                    out.write('[S] ' + value_type.group(1) + ' [R] value type [O] '+ value_type.group(2))
                                    out.write('\n')
                                else:
                                    if 'include []' in sent:
                                        possible_values = None
                                    else:
                                        possible_values = re.match(r'the possible values for \"([^\"]+)\" include (.+)', sent)
                                        
                                    if possible_values:
                                        pvs = possible_values.group(2)[:-1].split('[PLACEHOLDER]')
                                        for pv in pvs:
                                            out.write('[S] ' + possible_values.group(1) + ' [R] possible value [O] '+ pv.replace('[','').replace(']','').replace('\"','').strip())
                                            out.write('\n')
                                    else:
                                        default_value = re.match(r'the argument "([^\"]+)\" is .+ default value is set to (.+)', sent)
                                        if default_value:
                                            out.write('[S] ' + default_value.group(1) + ' [R] default value [O] '+ default_value.group(2).strip('.'))
                                            out.write('\n')
                                        else:
                                            records.append(sent)
        except Exception as error_trace:
            print(error_trace)
            print(api)
    out.close()
    file = open(os.path.join(args.output_dir,'constraints.txt'),'r').readlines()
    
    level1=None
    level2=None
    level1s = []
    level2s = []
    constraints_mapping = {}
    dt=[]
    para_name_list = []
    function_constraints = {}
    for line in file:
        if '>>>>>>' in line:
            level2=line.split('>>>>>> ')[1].strip()
            constraints_mapping[level1][level2]={}
            level2s.append(level2)
            function_constraints[level1+'.'+level2]=['[S] ' + level1 + ' [R] has function [O] '+level2]
        elif '>>>' in line:
            level1=line.split('>>> ')[1].strip()
            if level1 not in constraints_mapping:
                constraints_mapping[level1]={}
                level1s.append(level1)
        else:
            key = line.split(' [R] ')[0].split('[S] ')[1]
            obj = line.split(' [O] ')[1].strip()
            if key != level1:
                function_constraints[level1+'.'+level2].append(line.strip())
                if key == level2:
                    constraints_mapping[level1][level2][obj]=[line.strip()]
                else:
                    constraints_mapping[level1][level2][key].append(line.strip())
                    para_name_list.append(key)
                    if 'value type' in line:
                        dt.append(obj)

    api_doc = {}
    for level1 in tqdm(constraints_mapping):
        api_doc[level1]={}
        for level2 in constraints_mapping[level1]:
            api_doc[level1][level2]={}
            for para in constraints_mapping[level1][level2]:
                api_doc[level1][level2][para]={}
                pv_list = []
                for cs in constraints_mapping[level1][level2][para]:
                    if '[R] required parameter [O]' in cs:
                        api_doc[level1][level2][para]['required']=1
                    elif '[R] optional parameter [O]' in cs:
                        api_doc[level1][level2][para]['required']=0
                    elif '[R] possible value [O]' in cs:
                        pv_list.append(cs.split('[R] possible value [O]')[-1].strip())
                if pv_list:
                    api_doc[level1][level2][para]['pv_list']=pv_list
    
    for pack in api_doc:
        for func in api_doc[pack]:
            api_doc[pack][func]['para_pv_ids']={}
            api_doc[pack][func]['required_para']=[]
            api_doc[pack][func]['all_para']=[]
            
            for para in api_doc[pack][func]:
                api_doc[pack][func]['all_para'].append(para)
                if para not in ['para_pv_ids','required_para','all_para'] and api_doc[pack][func][para]['required']:
                    api_doc[pack][func]['required_para'].append(para)
                
                if 'pv_list' in api_doc[pack][func][para]:
                    cq = [getTokenIds(pv)+[CLOSE_QUOTE] for pv in api_doc[pack][func][para]['pv_list']]
                    cqc = [getTokenIds(pv)+[CLOSE_QUOTE_COMMA] for pv in api_doc[pack][func][para]['pv_list']]
                    cqrb = [getTokenIds(pv)+[CLOSE_QUOTE_RIGHT_BRACKET] for pv in api_doc[pack][func][para]['pv_list']]
                    api_doc[pack][func]['para_pv_ids'][para]=construct_tree(cq+cqc+cqrb)
            
            all_para_ids = [getTokenIds(para)+[EQUAL_OPEN_QUOTE] for para in api_doc[pack][func]['all_para']]
            all_para_ids_with_space = [getTokenIdsWithLeadingSpace(para)+[EQUAL_OPEN_QUOTE] for para in api_doc[pack][func]['all_para']]
            
            api_doc[pack][func]['para_gen_options'] = construct_tree(all_para_ids_with_space)
            api_doc[pack][func]['first_para_gen_options'] = construct_tree(all_para_ids+[[RIGHT_BRACKET]])
            
            required_ids_with_space = [getTokenIdsWithLeadingSpace(rq_para)+[EQUAL_OPEN_QUOTE] for rq_para in api_doc[pack][func]['required_para']]
            api_doc[pack][func]['required_ids-with_space'] = construct_tree(required_ids_with_space)

    pk.dump(api_doc, open(os.path.join(args.output_dir,'api_constraints-tree.pk'),'wb'))
    lists = [getTokenIds(pack)+[DOT] for pack in list(api_doc.keys())]
    package_pv = construct_tree(lists)
    pk.dump(package_pv, open(os.path.join(args.output_dir,'api_constraints-package_pv-tree.pk'),'wb'))
    
    lists = [getTokenIdsWithLeadingSpace(pack)+[DOT] for pack in list(api_doc.keys())]
    package_pv = construct_tree(lists)
    pk.dump(package_pv, open(os.path.join(args.output_dir,'api_constraints-package_pv_leading_space-tree.pk'),'wb'))
    
    func_pv={}
    for pack in list(api_doc.keys()):
        lists = [getTokenIds(func)+[LEFT_BRACKET] for func in api_doc[pack]]
        func_pv[pack] = construct_tree(lists)
    pk.dump(func_pv, open(os.path.join(args.output_dir,'api_constraints-function_pv-tree.pk'),'wb'))

print('Success!')