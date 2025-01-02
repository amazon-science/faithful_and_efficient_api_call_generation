#!/usr/bin/env python
# coding: utf-8

from transformers import AutoTokenizer, LlamaForCausalLM, StoppingCriteria, StoppingCriteriaList
import torch
from tqdm import tqdm
import json
import sys
import pickle as pk
from datasets import Dataset 
import os
from accelerate import infer_auto_device_map
import numpy as np
import argparse
import copy
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True, help="")
parser.add_argument("--dataset_path", required=True, help="")
parser.add_argument("--output_file", required=True, help="")
parser.add_argument("--gpu_device", default='0', type=str,
                help="GPU device id")
parser.add_argument("--config_file", default=None, type=str,
                help="(optional) loading arguments with config file")
parser.add_argument("--constraints_path", default=None, type=str,
                help="Path to folder that hosts the constraints files")
parser.add_argument("--beam_size", default=4, type=int,
                help="Size of the beam search")
parser.add_argument("--topkp", action='store_true',
                help="Use top-k or top-p sampling")
parser.add_argument("--k", default=None, type=int,
                help="Value of k for top-k sampling, use 1 for greedy search")
parser.add_argument("--p", default=0.9, type=float,
                help="Value of p for top-p sampling, default to 0.9, will be overrided if k is set")
parser.add_argument("--max_new_tokens", default=200, type=int,
                help="Maximum amount of tokens to generation, default to 200")

args = parser.parse_args()

if args.config_file != None:
    with open(args.config_file, 'r') as f:
        args.__dict__ = json.load(f)
        
output_dir = os.path.dirname(args.output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(args.__dict__)
with open(os.path.join(output_dir,'config.dict'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

model_records = {
                 'alpaca-13b-gpt4':'chavinlo/gpt4-x-alpaca',
                 }

model_path = args.model_path
if model_path in model_records:
    model_path = model_records[model_path]

dataset_records = {
                   'dstc8':'$ We use an amazon version of dstc8 data repurposed for api call generation. Due to ownership of the data, we only provide a sample input in this repo$', 
                   }

data_path = args.dataset_path
if data_path in dataset_records:
    data_path = dataset_records[data_path]
    
device = torch.device('cuda:'+str(args.gpu_device)) if torch.cuda.is_available() else torch.device('cpu')
device_mapping = 'balanced_low_0' # auto / balanced_low_0

model = LlamaForCausalLM.from_pretrained(model_path, device_map=device_mapping)

tokenizer = AutoTokenizer.from_pretrained(model_path)

if data_path.endswith('.arrow'):
    raw_data = Dataset.from_file(data_path)
else:
    try:
        raw_data = json.load(open(data_path,'r'))
    except:
        try:
            raw_data = json.loads(open(data_path,'r').readline())
        except:
            raw_data = json.loads(' '.join(open(data_path,'r').readlines()))    

prompts = []
for sample in tqdm(raw_data):
    processed_prompts = '\n'.join([sample['instruction'].strip(),'\nInput:', sample['input']])
    processed_prompts = prompts.append((processed_prompts,sample['expected_output']))

DOT = tokenizer.convert_tokens_to_ids('.')
LEFT_BRACKET = tokenizer.convert_tokens_to_ids('(')
RIGHT_BRACKET = tokenizer.convert_tokens_to_ids(')')
EQUAL_OPEN_QUOTE = tokenizer.convert_tokens_to_ids('="')
CLOSE_QUOTE_COMMA = tokenizer.convert_tokens_to_ids('",')
CLOSE_QUOTE_RIGHT_BRACKET = tokenizer.convert_tokens_to_ids('")')
CLOSE_QUOTE = tokenizer.convert_tokens_to_ids('"')
COMMA = tokenizer.convert_tokens_to_ids(',')

DOT_CLOSE_QUOTE_RIGHT_BRACKET = 23157
DOT_CLOSE_QUOTE_COMMA = 19602
api_doc = pk.load(open(os.path.join(args.constraints_path,'api_constraints-tree.pk'),'rb'))
package_pv = pk.load(open(os.path.join(args.constraints_path,'api_constraints-package_pv-tree.pk'),'rb'))
package_pv_leading_space = pk.load(open(os.path.join(args.constraints_path,'api_constraints-package_pv_leading_space-tree.pk'),'rb'))
func_pv = pk.load(open(os.path.join(args.constraints_path,'api_constraints-function_pv-tree.pk'),'rb'))

def bfs_constrained_beam_search_unified(initial_beams, beam_size, input_length, max_length):
    global estimated_time_savings
    
    current_level_beams = initial_beams
    
    while True:
        
        termination_flag = True
        next_level_beams = []
        
        for beam, node, meta, model_kwargs in current_level_beams:

            last_output_id = beam[0][-1].item()

            if beam.size()[-1]-input_length > max_length:
                next_level_beams.append((beam, node, meta, model_kwargs))
                continue
            else:
                if last_output_id in [RIGHT_BRACKET,CLOSE_QUOTE_RIGHT_BRACKET]:
                    required_para = api_doc[meta['package_name']][meta['function_name']]['required_para']
                    if len(set(required_para)-set(meta['decoded_para']))>0:
                        
                        termination_flag = False
                        
                        beam[0][-1] = torch.tensor(CLOSE_QUOTE_COMMA)
                        node = api_doc[meta['package_name']][meta['function_name']]['required_ids-with_space']                        
                        
                    else:
                        next_level_beams.append((beam, node, meta, model_kwargs))
                        continue

                elif last_output_id==LEFT_BRACKET:
                    
                    termination_flag = False
                    
                    node = api_doc[meta['package_name']][meta['function_name']]['para_gen_options']

                elif last_output_id==EQUAL_OPEN_QUOTE:
                    
                    termination_flag = False

                    parameter_name = tokenizer.decode(beam[0][input_length:-1]).split('(')[-1].split(', ')[-1]

                    meta = copy.deepcopy(meta)
                    meta['decoded_para'].append(parameter_name)

                    parameter_with_pv = api_doc[meta['package_name']][meta['function_name']]['para_pv_ids']

                    if parameter_name in parameter_with_pv:
                        node = parameter_with_pv[parameter_name]
                    else:
                        node = None

                elif last_output_id in [CLOSE_QUOTE, CLOSE_QUOTE_COMMA]:
                    
                    termination_flag = False

                    node = api_doc[meta['package_name']][meta['function_name']]['para_gen_options']
                    
            if node:
                next_candidates = list(node.keys())

                if len(next_candidates) == 0:
                    next_level_beams.append((beam, node, meta, model_kwargs))
                    continue
                    
                else:

                    termination_flag = False

                    if len(next_candidates) == 1:
                        
                        # >>> This inference step can be removed for packages supporting partial past_key_values. Added estimated_time_savings for the compatibility and efficiency of experimenting with HF transformers package
            
                        torch.cuda.synchronize()
                        timepoint = time.perf_counter()
                        
                        model_kwargs = copy.deepcopy(model_kwargs)
                        model_inputs = model.prepare_inputs_for_generation(beam, **model_kwargs)

                        with torch.no_grad():
                            outputs = model(**model_inputs, return_dict=True)

                        model_kwargs = model._update_model_kwargs_for_generation(
                                    outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
                                )

                        torch.cuda.synchronize()
                        estimated_time_savings += time.perf_counter()-timepoint
                        # <<< This inference step can be removed for packages supporting partial past_key_values. Added estimated_time_savings for the compatibility and efficiency of experimenting with HF transformers package

                        next_token_id = torch.tensor([[next_candidates[0]]]).to(device)
                        new_beam = torch.cat([beam, next_token_id], dim=-1)
                        new_node = node[next_candidates[0]]

                        next_level_beams.append((new_beam, new_node, meta, model_kwargs))

                    else:

                        model_kwargs = copy.deepcopy(model_kwargs)
                        model_inputs = model.prepare_inputs_for_generation(beam, **model_kwargs)

                        with torch.no_grad():
                            outputs = model(**model_inputs, return_dict=True)

                        model_kwargs = model._update_model_kwargs_for_generation(
                                    outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
                                )
                        
                        next_token_logits = outputs.logits[:, -1, :]
                        next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)

                        selected_prob = next_token_log_probs[0][next_candidates]
                        selected_prob = selected_prob.tolist()

                        if len(next_candidates) > beam_size:
                            sorted_pairs = sorted(list(zip(next_candidates, selected_prob)), key=lambda x: x[1], reverse=True)
                            next_candidates, selected_prob = zip(*sorted_pairs)
                            next_candidates = next_candidates[:beam_size]
                            selected_prob = selected_prob[:beam_size]

                        for next_token_id, next_token_score in zip(next_candidates, selected_prob):
                            new_beam = torch.cat([beam, torch.tensor([[next_token_id]]).to(device)], dim=-1)
                            new_meta = meta.copy()
                            new_meta['score'] += next_token_score
                            new_node = node[next_token_id]
                            next_level_beams.append((new_beam, new_node, new_meta, model_kwargs))
            else:
                last_output_id = beam[0][-1].item()

                if last_output_id in [CLOSE_QUOTE_COMMA, CLOSE_QUOTE_RIGHT_BRACKET]:
                
                    next_level_beams.append((beam, node, meta, model_kwargs))
                    continue

                else:
                    termination_flag = False

                    model_kwargs = copy.deepcopy(model_kwargs)
                    model_inputs = model.prepare_inputs_for_generation(beam, **model_kwargs)

                    with torch.no_grad():
                        outputs = model(**model_inputs, return_dict=True)

                    model_kwargs = model._update_model_kwargs_for_generation(
                                outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
                            )

                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)

                    topk_log_probs, topk_ids = torch.topk(next_token_log_probs, beam_size)

                    for i in range(beam_size):

                        next_token_id = topk_ids[0, i].unsqueeze(0).unsqueeze(0)
                        log_prob = topk_log_probs[0, i].item()

                        if next_token_id.item() == DOT_CLOSE_QUOTE_RIGHT_BRACKET:
                            new_beam = torch.cat([beam, torch.tensor([[DOT, CLOSE_QUOTE_RIGHT_BRACKET]]).to(device)], dim=-1)
                        elif next_token_id.item() == DOT_CLOSE_QUOTE_COMMA:
                            new_beam = torch.cat([beam, torch.tensor([[DOT, CLOSE_QUOTE_COMMA]]).to(device)], dim=-1)
                        else:
                            new_beam = torch.cat([beam, next_token_id], dim=-1)

                        new_meta = meta.copy()
                        new_meta['score'] += log_prob
                        next_level_beams.append((new_beam, node, new_meta, model_kwargs))
        
        current_level_beams = sorted(next_level_beams, key=lambda x: x[2]['score'], reverse=True)[:beam_size]
        
        if termination_flag:
            break
            
    return [beam[:,input_length:] for beam, _, _, _ in current_level_beams]


def bfs_constrained_beam_search_decoding_of_a_unit(initial_beam, initial_node, initial_meta, beam_size, model_kwargs={}):
    global estimated_time_savings
    
    current_level_beams = [(initial_beam, initial_node, initial_meta, model_kwargs)]
    
    while True:
        
        termination_flag = True
        next_level_beams = []
        
        for beam, node, meta, model_kwargs in current_level_beams:
            
            next_candidates = list(node.keys())
            
            if len(next_candidates) == 0:
                next_level_beams.append((beam, node, meta, model_kwargs))
                continue
            
            termination_flag = False
            
            if len(next_candidates) == 1:
                
                # >>> This inference step can be removed for packages supporting partial past_key_values. Added estimated_time_savings for the compatibility and efficiency of experimenting with HF transformers package
            
                torch.cuda.synchronize()
                timepoint = time.perf_counter()
                
                model_kwargs = copy.deepcopy(model_kwargs)
                model_inputs = model.prepare_inputs_for_generation(beam, **model_kwargs)

                with torch.no_grad():
                    outputs = model(**model_inputs, return_dict=True)

                model_kwargs = model._update_model_kwargs_for_generation(
                            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
                        )

                torch.cuda.synchronize()
                estimated_time_savings += time.perf_counter()-timepoint
                # <<< This inference step can be removed for packages supporting partial past_key_values. Added estimated_time_savings for the compatibility and efficiency of experimenting with HF transformers package
                
                next_token_id = torch.tensor([[next_candidates[0]]]).to(device)
                new_beam = torch.cat([beam, next_token_id], dim=-1)
                new_node = node[next_candidates[0]]
                
                next_level_beams.append((new_beam, new_node, meta, model_kwargs))
                
            else:
                
                model_kwargs = copy.deepcopy(model_kwargs)
                model_inputs = model.prepare_inputs_for_generation(beam, **model_kwargs)

                with torch.no_grad():
                    outputs = model(**model_inputs, return_dict=True)

                model_kwargs = model._update_model_kwargs_for_generation(
                            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
                        )
                
                next_token_logits = outputs.logits[:, -1, :]
                next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)
                    
                selected_prob = next_token_log_probs[0][next_candidates]
                selected_prob = selected_prob.tolist()
                
                if len(next_candidates) > beam_size:
                    sorted_pairs = sorted(list(zip(next_candidates, selected_prob)), key=lambda x: x[1], reverse=True)
                    next_candidates, selected_prob = zip(*sorted_pairs)
                    next_candidates = next_candidates[:beam_size]
                    selected_prob = selected_prob[:beam_size]
                    
                for next_token_id, next_token_score in zip(next_candidates, selected_prob):
                    new_beam = torch.cat([beam, torch.tensor([[next_token_id]]).to(device)], dim=-1)
                    new_meta = meta.copy()
                    new_meta['score'] += next_token_score
                    new_node = node[next_token_id]
                    next_level_beams.append((new_beam, new_node, new_meta, model_kwargs))
            
        current_level_beams = sorted(next_level_beams, key=lambda x: x[2]['score'], reverse=True)[:beam_size]
        
        if termination_flag:
            break
            
    return [(beam, meta, model_kwargs) for beam, node, meta, model_kwargs in current_level_beams]

def constrained_beam_search(input_text, beam_size, max_length):
    global package_function_beams
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    meta = {'score':0, 'package_name':None, 'function_name':None, 'decoded_para':[]}
    input_length = input_ids.size()[-1]

    package_beams = bfs_constrained_beam_search_decoding_of_a_unit(input_ids, package_pv, meta, beam_size=beam_size, model_kwargs={})
    package_function_beams = []
    for package_beam, meta, model_kwargs in package_beams:
        package_name = tokenizer.decode(package_beam[0][input_length:])[:-1]
        meta['package_name'] = package_name
        node = func_pv[package_name]
        package_function_beams += bfs_constrained_beam_search_decoding_of_a_unit(package_beam, node, meta, beam_size=beam_size, model_kwargs=model_kwargs)
                
    package_function_beams = sorted(package_function_beams, key=lambda x: x[1]['score'], reverse=True)[:1]
    
    for beam, meta, model_kwargs in package_function_beams:
        if not meta['function_name']:
            function_name = tokenizer.decode(beam[0][input_length:]).split('.')[-1][:-1]
            meta['function_name'] = function_name
        
    initial_beams = [(pfb[0],None,pfb[1],pfb[2]) for pfb in package_function_beams]
    beams = bfs_constrained_beam_search_unified(initial_beams, beam_size, input_length, max_length)
    
    return beams

def topkp_constrained_decoding_of_a_unit(output_ids, node, k=None, model_kwargs = {}):
    global estimated_time_savings
    next_candidates = list(node.keys())
    unit = []
    while len(next_candidates)>0:
        if len(next_candidates)==1:
            
            # >>> This inference step can be removed for packages supporting partial past_key_values. Added estimated_time_savings for the compatibility and efficiency of experimenting with HF transformers package
            
            torch.cuda.synchronize()
            timepoint = time.perf_counter()
            
            model_inputs = model.prepare_inputs_for_generation(output_ids, **model_kwargs)
        
            with torch.no_grad():
                outputs = model(**model_inputs, return_dict=True)

            model_kwargs = model._update_model_kwargs_for_generation(
                        outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
                    )
            
            torch.cuda.synchronize()
            estimated_time_savings += time.perf_counter()-timepoint
            # <<< This inference step can be removed for packages supporting partial past_key_values. Added estimated_time_savings for the compatibility and efficiency of experimenting with HF transformers package
            
            unit.append(next_candidates[0])
            next_token_id = torch.tensor([[next_candidates[0]]]).to(device)
            output_ids = torch.cat([output_ids, next_token_id], dim=-1)
            
            node = node[next_candidates[0]]
            next_candidates = list(node.keys())
        else:
            
            model_inputs = model.prepare_inputs_for_generation(output_ids, **model_kwargs)

            with torch.no_grad():
                outputs = model(**model_inputs, return_dict=True)

            model_kwargs = model._update_model_kwargs_for_generation(
                        outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
                    )
                
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
        
            selected_prob = next_token_probs[0][next_candidates]
            
            if k:
                topk_probs, topk_idxs = torch.topk(selected_prob, k)
                token_key = next_candidates[topk_idxs[torch.multinomial(topk_probs,1)]]
            else:
                token_key = next_candidates[torch.multinomial(selected_prob,1)]
            
            unit.append(token_key)
            next_token_id = torch.tensor([[token_key]]).to(device)
            output_ids = torch.cat([output_ids, next_token_id], dim=-1)
            
            node = node[token_key]
            next_candidates = list(node.keys())
    
    return output_ids, unit, model_kwargs


def constrained_topkp_sampling(input_ids, k=None, p=0.9, max_length=200, input_length=None):
    output_ids = input_ids

    model_kwargs = {}
    
    if not input_length:
        input_length = output_ids.size()[-1]
            
    model_kwargs = {}

    output_ids, function_unit, model_kwargs = topkp_constrained_decoding_of_a_unit(output_ids, func_pv, k=k, model_kwargs=model_kwargs)        
    package_name = tokenizer.decode(package_unit).strip()[:-1]

    node = func_pv[package_name]
    output_ids, function_unit, model_kwargs = constrained_decoding_of_a_unit(output_ids, node, k=k, model_kwargs=model_kwargs)
    function_name = tokenizer.decode(function_unit).split('.')[-1].strip()[:-1]

    decoded_para = []
    required_para = api_doc[package_name][function_name]['required_para']
    parameter_with_pv = api_doc[package_name][function_name]['para_pv_ids']

    unit=None
    last_output_id = output_ids[0][-1].item()
    
    closing_symbols = []
    
    while last_output_id not in [tokenizer.eos_token_id]:
        if output_ids.size()[-1]-input_length >= max_length:
            break

        if last_output_id==LEFT_BRACKET:
            gen_options = api_doc[package_name][function_name]['para_gen_options']
            output_ids, unit, model_kwargs = constrained_decoding_of_a_unit(output_ids, gen_options, k=k, model_kwargs=model_kwargs)
            
        elif last_output_id == EQUAL_OPEN_QUOTE:            
            parameter_name = tokenizer.decode(unit[:-1])
            decoded_para.append(parameter_name)
                
            if parameter_name in parameter_with_pv:
                gen_options = parameter_with_pv[parameter_name]
                output_ids, unit, model_kwargs = constrained_decoding_of_a_unit(output_ids, gen_options, k=k, model_kwargs=model_kwargs)
            else:
                while last_output_id not in [CLOSE_QUOTE_COMMA, CLOSE_QUOTE_RIGHT_BRACKET]:
                    model_inputs = model.prepare_inputs_for_generation(output_ids, **model_kwargs)
        
                    with torch.no_grad():
                        outputs = model(**model_inputs, return_dict=True)

                    model_kwargs = model._update_model_kwargs_for_generation(
                                outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
                            )
                    
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_probs = torch.softmax(next_token_logits, dim=-1)
                    
                    if k:
                        topk_probs, topk_idxs = torch.topk(next_token_probs, k)
                        next_token_id = topk_idxs.squeeze()[torch.multinomial(topk_probs,1)]
                    else:
                        sorted_probs, sorted_indices = torch.sort(next_token_probs, descending=True)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                        selected_indices = sorted_indices[cumulative_probs <= p]
                        if len(selected_indices) == 0:
                            selected_indices = sorted_indices[:, 0]
                        next_token_id = selected_indices[torch.multinomial(sorted_probs[:,:len(selected_indices)],1)]
                    
                    if next_token_id.item() == DOT_CLOSE_QUOTE_RIGHT_BRACKET:
                        output_ids = torch.cat([output_ids, torch.tensor([[DOT,CLOSE_QUOTE_RIGHT_BRACKET]]).to(device)], dim=-1)
                    elif next_token_id.item() ==  DOT_CLOSE_QUOTE_COMMA:
                        output_ids = torch.cat([output_ids, torch.tensor([[DOT,CLOSE_QUOTE_COMMA]]).to(device)], dim=-1)
                    else:
                        output_ids = torch.cat([output_ids, next_token_id.flatten().view(1,1)], dim=-1)
                    
                    last_output_id = output_ids[0][-1].item()
                    
                    if output_ids.size()[-1]-input_length >= max_length:
                        break
        elif last_output_id in [CLOSE_QUOTE, CLOSE_QUOTE_COMMA]:
            gen_options=api_doc[package_name][function_name]['para_gen_options']
            output_ids, unit, model_kwargs = constrained_decoding_of_a_unit(output_ids, gen_options, k=k, model_kwargs=model_kwargs)
        elif last_output_id in [RIGHT_BRACKET,CLOSE_QUOTE_RIGHT_BRACKET]:
            if len(set(required_para) - set(decoded_para))>0:
                output_ids[0][-1] = torch.tensor(CLOSE_QUOTE_COMMA)
                new_options = api_doc[package_name][function_name]['para_gen_options']
                output_ids, unit, model_kwargs = constrained_decoding_of_a_unit(output_ids, new_options, k=k, model_kwargs=model_kwargs)
            else:
                break
        last_output_id = output_ids[0][-1].item()

    return output_ids

estimated_time_savings=0
decoding_records = []
for p in tqdm(prompts):
    if args.topkp:
        output_ids = constrained_topkp_sampling(tokenizer.encode(p[0], return_tensors='pt').to(device),k=args.k, p=args.p, max_length=args.max_new_tokens)
        processed_results = tokenizer.batch_decode(output_ids)[0].split('Generate API Request:\n')[-1].strip()
        decoding_records.append((p, processed_results))
    else:
        result_beams = constrained_beam_search(p[0], beam_size=args.beam_size, max_length=args.max_new_tokens)
        decoding_records.append((p, [tokenizer.batch_decode(b)[0] for b in result_beams]))
print(estimated_time_savings)
pk.dump(decoding_records,open('decoding_records.pk','wb'))
