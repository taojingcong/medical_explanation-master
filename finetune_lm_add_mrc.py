# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modifications copyright (c) 2020 Sawan Kumar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modification history
# 2020 Sawan Kumar: Modified to finetune and generate explanations for NLI
# Modification history
# 2020 Dongfangli: Modified to finetune and generate explanations for Medical QA
# 2020 Dongfangli: Add MRC model and KL divergence loss

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import pandas as pd

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                                  get_cosine_with_hard_restarts_schedule_with_warmup,
                                  BertConfig, BertLMHeadModel, BertTokenizer,
                                  AlbertConfig, AlbertTokenizer, RobertaConfig, RobertaTokenizer, RobertaModel, RobertaForMultipleChoice,
                                GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
from transformers import BertForMultipleChoice

# from transformers.modeling_bert import BertLMHeadModel
from model import BertLMAddMrcHeadModel, AlbertLMAddMrcHeadModel, RobertaLMAddMrcHeadModel, GPT2LMAddMrcHead, AlbertOneLMAddMrcHeadModel
from lm_utils import CLS_TOKEN, SEP_TOKEN, TSVAddMRCDataset, EOS_TOKEN, GPTTSVAddMRCDataset
# EXP_TOKEN, EOS_TOKEN

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    # 'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'bert': (BertConfig, BertLMAddMrcHeadModel, BertTokenizer),
    'albert': (AlbertConfig, AlbertLMAddMrcHeadModel, AlbertTokenizer),
    'albert-only': (AlbertConfig, AlbertOneLMAddMrcHeadModel, AlbertTokenizer),
    'roberta': (RobertaConfig, RobertaLMAddMrcHeadModel, RobertaTokenizer),
    'gpt': (GPT2Config, GPT2LMAddMrcHead, GPT2Tokenizer)
}

cross_entropy_ignore_index = 0

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

# def clip_batch(batch):
#     """
#     """
#     # print("batch size is {}".format(len(batch[0])))
#     idx, input_ids, attention_mask, token_type_ids, labels = batch
#     # [batch_size, 2, L]
#     batch_size = input_ids.size(0)
#     while True:
#         end_flag = False
#         for i in range(batch_size):
#             if input_ids[i, 0, -1] != 0:
#                 end_flag = True
#             if input_ids[i, 1, -1] != 0:
#                 end_flag = True 
        
#         if end_flag:
#             break
#         else:
#             input_ids = input_ids[:, :, :-1]
    
#     max_seq_length = input_ids.size(2)
#     attention_mask = attention_mask[:, :, :max_seq_length]
#     token_type_ids = token_type_ids[:, :, :max_seq_length]
    
#     return idx, input_ids, attention_mask, token_type_ids, labels 

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    # warmup_proportion * t_total
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps*t_total, num_training_steps=t_total) 
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_accuracy = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            #batch(问题+选项+解释)[16,65],batch_mrc(option_inputs)[16,5,65],batch_mask_mrc(option_inputs的mask)[16,5,65],batch_segment_mrc[16,5,65],promput_length(问题+选项的长度)[16,],
            #total_length(option_input的长度)[16,],label_mrc正确选项[16,]
            batch, batch_mrc, batch_mask_mrc, batch_segment_mrc, prompt_lengths, total_lengths, labels_mrc = batch  
            max_length = torch.max(total_lengths).item()#option_input的最大长度
            batch = batch[:, :max_length]#batch=16*58
            # max_length = args.block_size
            # 设batch中最长句子的长度为max_seq_length, 将超过max_seq_length的部分删除
            inputs, labels = (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            batch_size = batch_mrc.size(0)

            #下面这段循环好像没意义?
            while True:
                end_flag = False
                for i in range(batch_size):
                    if batch_mrc[i, 0, -1] != 0:
                        end_flag = True
                    if batch_mrc[i, 1, -1] != 0:
                        end_flag = True 
                
                if end_flag:
                    break
                else:
                    batch_mrc = batch_mrc[:, :, :-1]
            
            max_seq_length = batch_mrc.size(2)
            batch_mask_mrc = batch_mask_mrc[:, :, :max_seq_length]
            batch_segment_mrc = batch_segment_mrc[:, :, :max_seq_length]
    
            labels_mrc = labels_mrc.to(args.device)
            batch_mrc = batch_mrc.to(args.device)
            batch_mask_mrc = batch_mask_mrc.to(args.device)
            batch_segment_mrc = batch_segment_mrc.to(args.device)
            total_lengths = torch.tensor(total_lengths)
            #这一步是干啥的?
            attention_mask = torch.arange(max_length).expand(len(total_lengths), max_length) < total_lengths.unsqueeze(1)
            attention_mask = attention_mask[:, :max_length]
            attention_mask = attention_mask.to(args.device)
            # total_lengths = total_lengths.to(args.device)
            # for idx in range(len(prompt_lengths)):
            #     labels[idx, :prompt_lengths[idx]] = cross_entropy_ignore_index
            # print(attention)
            model.train()
            #input_ids为option_inputs[16,58] ,attention_mask[16,58], attention_mask_mrc[16,5,41]不知道根据啥截断的,token_type_ids_mrc[16,5,41],labels原始option_input的序列[16,58]
            outputs = model(input_ids=inputs, attention_mask=attention_mask, attention_mask_mrc=batch_mask_mrc, token_type_ids_mrc=batch_segment_mrc,\
                    input_ids_mrc=batch_mrc, labels=labels, labels_mrc=labels_mrc)
            # loss = outputs[0] # model outputs are always tuple in transformers (see doc)
            # print(outputs[0], outputs[1], outputs[2])
            # mse mrc lm
            loss = 0.3 * outputs[0] + 0.6 * outputs[1] + 0.1 * outputs[2] # add two loss TODO 不同的权重
            # loss = outputs[1]
            # loss = outputs[1] + outputs[2]
            
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                           tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss
                assert args.save_steps == args.logging_steps, "Save steps must equal to logging steps."
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    print("{} {}".format(results['mrc_accuracy'], best_accuracy))
                    if results['mrc_accuracy'] > best_accuracy:
                        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)
                        best_accuracy = results['mrc_accuracy']
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
        # result = evaluate(args, model, tokenizer)
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step
 

def sample_sequence(model, length, context, mrc_context, batch_mask_mrc, batch_segment_mrc, device='cpu', sep_token_id=None):
    context = torch.tensor(context, dtype=torch.long, device=device)
    mrc_context = torch.tensor(mrc_context, dtype=torch.long, device=device)
    batch_mask_mrc = torch.tensor(batch_mask_mrc, dtype=torch.long, device=device)
    batch_segment_mrc = torch.tensor(batch_segment_mrc, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    print(generated)
    past = None
    # length = 1
    with torch.no_grad():
        for i in range(length):
            #inputs = {'input_ids': context}
            #output, past = model(**inputs, past=past)
            input_shape = generated.shape
            attention_mask = generated.new_ones(input_shape)
            generated_r = model.prepare_inputs_for_generation(generated, attention_mask)
            inputs = {'input_ids': generated, "attention_mask": None,\
                "attention_mask_mrc": batch_mask_mrc, "token_type_ids_mrc": batch_segment_mrc, \
                "input_ids_mrc": mrc_context, "return_dict": False}
            
            output= model(**inputs)[0] 
            # print(output)
            next_token_logits = output[0, -1, :]
            next_token = torch.argmax(next_token_logits)
            generated = torch.cat((generated, next_token.view(1,1)), dim=1)
            if next_token.item() == sep_token_id:
                break
            context = next_token.view(1,1)
    return generated


def generate(args, model, tokenizer, prefix=""):
    if args.length < 0 and model.config.max_position_embeddings > 0:
        args.length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.length:
        args.length = model.config.max_position_embeddings  # No generation bigger than model size 
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    eval_output_dir = args.output_dir
    if args.model_type == "gpt":
        eval_dataset = GPTTSVAddMRCDataset(tokenizer, args, file_path=args.eval_data_file,
                                block_size=args.block_size, get_annotations=False)
    else:
        eval_dataset = TSVAddMRCDataset(tokenizer, args, file_path=args.eval_data_file,
                                block_size=args.block_size, get_annotations=False)

    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)

    # Eval!
    logger.info("***** Running generation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))

    model.eval()

    for index, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
        batch, batch_mrc, inputs_mask_mrc, inputs_segment_mrc, prompt_lengths, total_lengths, labels_mrc = batch
        batch = batch.squeeze()
        batch_mrc = batch_mrc.squeeze()
        inputs_mask_mrc = inputs_mask_mrc.squeeze()
        inputs_segment_mrc = inputs_segment_mrc.squeeze()
        out = sample_sequence(
            model=model,
            context=batch,
            mrc_context=batch_mrc,
            batch_mask_mrc=inputs_mask_mrc,
            batch_segment_mrc=inputs_segment_mrc,
            length=args.length,
            device=args.device,
            sep_token_id=tokenizer.convert_tokens_to_ids(SEP_TOKEN),
        )
        out = out[0, len(batch):].tolist()
        text = tokenizer.decode(out, clean_up_tokenization_spaces=True)
        text = text.split(SEP_TOKEN)[0].strip()
        eval_dataset.add_explanation(index, text)
        print (text)

    #save
    directory, filename = os.path.split(args.eval_data_file)
    model_directory, model_name = os.path.split(os.path.normpath(args.output_dir))
    output_name = os.path.join(directory, '{}_{}'.format(model_name, filename))
    eval_dataset.save(output_name)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def evaluate(args, model, tokenizer, prefix=""):
    eval_output_dir = args.output_dir
    if args.model_type == "gpt":
        eval_dataset = GPTTSVAddMRCDataset(tokenizer, args, file_path=args.eval_data_file,
                                block_size=args.block_size, get_annotations=True)
    else:
        eval_dataset = TSVAddMRCDataset(tokenizer, args, file_path=args.eval_data_file,
                                    block_size=args.block_size, get_annotations=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    eval_accuracy = 0.0
    
    eval_mse_loss = 0.0
    eval_mrc_loss = 0.0
    nb_eval_steps, nb_eval_examples = 0, 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        # batch, batch_mrc, prompt_lengths, total_lengths, labels_mrc = batch
        batch, batch_mrc, batch_mask_mrc, batch_segment_mrc, prompt_lengths, total_lengths, labels_mrc = batch  
        max_length = torch.max(total_lengths).item()
        batch = batch[:, :max_length]
        batch_size = batch_mrc.size(0)
        while True:
            end_flag = False
            for i in range(batch_size):
                if batch_mrc[i, 0, -1] != 0:
                    end_flag = True
                if batch_mrc[i, 1, -1] != 0:
                    end_flag = True 
            
            if end_flag:
                break
            else:
                batch_mrc = batch_mrc[:, :, :-1]
        
        max_seq_length = batch_mrc.size(2)
        batch_mask_mrc = batch_mask_mrc[:, :, :max_seq_length]
        batch_segment_mrc = batch_segment_mrc[:, :, :max_seq_length]

        # max_length = args.block_size
        batch = batch.to(args.device)
        labels_mrc = labels_mrc.to(args.device)
        batch_mrc = batch_mrc.to(args.device)
        batch_mask_mrc = batch_mask_mrc.to(args.device)
        batch_segment_mrc = batch_segment_mrc.to(args.device)
        total_lengths = torch.tensor(total_lengths)
        attention_mask = torch.arange(max_length).expand(len(total_lengths), max_length) < total_lengths.unsqueeze(1)
        attention_mask = attention_mask[:, :max_length]
        attention_mask = attention_mask.to(args.device)
    
            
        with torch.no_grad():
            outputs = model(input_ids=batch, attention_mask=attention_mask, attention_mask_mrc=batch_mask_mrc, token_type_ids_mrc=batch_segment_mrc,\
                        input_ids_mrc=batch_mrc, labels=batch, labels_mrc=labels_mrc)
            # mse mrc lm
            lm_loss = outputs[2]
            mse_loss = outputs[0]
            mrc_loss = outputs[1]
            if args.model_type == "gpt":
                logits = outputs[5]
            else:
                logits = outputs[4]
            logits = logits.detach().cpu().numpy()
            label_ids = labels_mrc.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_mse_loss += mse_loss.mean().item()
            eval_mrc_loss += mrc_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1
        nb_eval_examples += batch.size(0)

    eval_loss = eval_loss / nb_eval_steps
    eval_mse_loss = eval_mse_loss / nb_eval_steps
    eval_mrc_loss = eval_mrc_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "mse_loss": eval_mse_loss,
        "mrc_loss": eval_mrc_loss,
        "mrc_accuracy": eval_accuracy,
        "perplexity": perplexity
    }

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the eval data file")
    parser.add_argument("--do_generate", action='store_true',
                        help="Whether to generate text on the eval data file")
    parser.add_argument("--length", type=int, default=100,
                        help="Length for generation")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.1, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0.1, type=float,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--data_type", default="tsv", type=str,
                        help="Dataset type")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    if args.eval_data_file is None and args.do_eval:
        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
                         "or remove the --do_eval argument.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, #do_lower_case=args.do_lower_case,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    
    # lm_config = BertConfig.from_pretrained("bert-base-uncased",
    #                                         cache_dir=args.cache_dir if args.cache_dir else None)
    # lm_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=args.do_lower_case,
    #                                         cache_dir=args.cache_dir if args.cache_dir else None)


    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    if args.do_train:
        #Additional tokens
        if args.model_type == "gpt":
            print ('#tokens',  len(tokenizer))
            new_tokens = [SEP_TOKEN, EOS_TOKEN]
            tokenizer.add_tokens(new_tokens)
            
            print ('#extended tokens', len(tokenizer))
        if args.model_type == "albert":    
            # model = model_class(,config)
            # lm_config.update({'is_decoder': True})
            # model.bert = model.bert.from_pretrained("bert-base-uncased", from_tf=bool('.ckpt' in args.model_name_or_path), config=lm_config,
            #                                     cache_dir=args.cache_dir if args.cache_dir else None)
            # model.gpt = model.gpt.from_pretrained("gpt2-medium", from_tf=bool('.ckpt' in args.model_name_or_path), config=gpt_config,
            #                                     cache_dir=args.cache_dir if args.cache_dir else None)
            # config.update({'is_decoder': False})
            model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
            # model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config,
            #                                     cache_dir=args.cache_dir if args.cache_dir else None)
            # model.
            # model.resize_token_embeddings(len(tokenizer))
            
        elif args.model_type == "gpt" or args.model_type == "albert-only":
            if args.model_type == "albert-only":
                config.update({'is_decoder': True})
            model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
            model.resize_token_embeddings(len(tokenizer))
        print(len(tokenizer))
        print(model)
        model.to(args.device)



    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        if args.model_type == "gpt":
            train_dataset = GPTTSVAddMRCDataset(tokenizer, args, file_path=args.train_data_file,
                                    block_size=args.block_size, get_annotations=True)
        else:
            train_dataset = TSVAddMRCDataset(tokenizer, args, file_path=args.train_data_file,
                                        block_size=args.block_size, get_annotations=True)

        if args.local_rank == 0:
            torch.distributed.barrier()
        print(type(model))
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned

        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        model = model_class.from_pretrained(args.output_dir)
        model.to(args.device)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        result = evaluate(args, model, tokenizer)
        print (result)

    #Generation
    if args.do_generate:
        model = model_class.from_pretrained(args.output_dir)
        model.to(args.device)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        generate(args, model, tokenizer) 

if __name__ == "__main__":
    main()
