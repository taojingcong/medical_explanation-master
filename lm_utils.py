import pandas as pd
import os
import pickle
import torch
import json
from torch.utils.data import DataLoader, Dataset

CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
PAD_TOKEN = '[PAD]'
EOS_TOKEN = '[EOS]'

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class TSVDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train', block_size=512, get_annotations=False):
        self.print_count = 0
        self.sep_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
        self.cls_token_id = tokenizer.convert_tokens_to_ids(CLS_TOKEN)
        self.pad_token_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
        cached_features_file, data = self.load_data(file_path, block_size)
        self.data = data

        if get_annotations: cached_features_file = cached_features_file + '_annotated'

        # if os.path.exists(cached_features_file):
        #     print ('Loading features from', cached_features_file)
        #     with open(cached_features_file, 'rb') as handle:
        #         self.examples = pickle.load(handle)
        #     return

        print ('Saving features from ', file_path, ' into ', cached_features_file) 

        def create_example(r):
            # text1 = '{} ? {}'.format( r['questions'], r['options']) # medical
            text1 = '{} {}'.format( r['questions'], r['options']) # cos-e
            tokenized_text1 = [self.cls_token_id] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text1)) + [self.sep_token_id]
            prompt_length = len(tokenized_text1)
            tokenized_text, total_length = tokenized_text1, len(tokenized_text1)
            if get_annotations:
                text2 = r['explanations']
                tokenized_text2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text2))
                tokenized_text = tokenized_text1 + tokenized_text2
                tokenized_text = tokenized_text + [self.sep_token_id]
                total_length = len(tokenized_text)
                if len(tokenized_text) > block_size:
                    tokenized_text = tokenized_text[:block_size]
                if len(tokenized_text) < block_size:
                    tokenized_text = tokenized_text + [self.pad_token_id] * (block_size-len(tokenized_text))
            if self.print_count > 0:
                print(len(tokenized_text))
                print ('example: ', text1 + text2 if get_annotations else text1)
                self.print_count = self.print_count - 1
                print("total_length: ", total_length)
            return (tokenized_text, prompt_length, total_length)

        self.examples = self.data.apply(create_example, axis=1).to_list()
        
        print ('Saving ', len(self.examples), ' examples')
        with open(cached_features_file, 'wb') as handle:
            pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item][0]), self.examples[item][1], self.examples[item][2]

    def get_example_text(self, index):
        return self.data['prompt'][index]

    def add_explanation(self, index, explanation):
        explanation_name = 'Generated_Explanation'
        self.data.at[self.data.index[index], explanation_name] = explanation

    def load_data(self, file_path, block_size):
        assert os.path.isfile(file_path)
        # data = pd.read_csv(file_path, sep='\t', index_col='pairID')
        data = pd.read_json(file_path)
        # print (data)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, 'cached_lm_{}_{}'.format(block_size, filename))
        return cached_features_file, data

    def save(self, filename):
        # self.data.to_csv(filename, sep='#')
        # self.data.to_json(filename, force_ascii=False)
        with open(filename, "w") as f:
            data = self.data.to_dict('records')
            for i, exp in enumerate(data):
                exp['Generated_Explanation'] = exp['Generated_Explanation'].replace(" ", "")
                f.write(json.dumps(exp, ensure_ascii=False)+"\n")

class TSVAddMRCDataset(TSVDataset):
    def __init__(self, tokenizer, args, file_path='train', block_size=512, get_annotations=False, is_training=False):
        self.print_count = 0
        self.token_between = ", "
        self.sep_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
        self.cls_token_id = tokenizer.convert_tokens_to_ids(CLS_TOKEN)
        self.pad_token_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
        cached_features_file, data = self.load_data(file_path, block_size)
        self.data = data#data 为9741条数据,格式为[问题,选项,答案,解释,背景文本]

        if get_annotations: cached_features_file = cached_features_file + '_annotated'

        # if os.path.exists(cached_features_file):
        #     print ('Loading features from', cached_features_file)
        #     with open(cached_features_file, 'rb') as handle:
        #         self.examples = pickle.load(handle)
        #     return

        # print ('Saving features from ', file_path, ' into ', cached_features_file) 

        def create_example(r):
            # text1 = '{} ? {}'.format( r['questions'], r['options']) # medical
            # cos-e :
            #r为一条数据
            #[问题,选项,解释,答案,options_input]
            options_text = self.token_between.join(r['options'].split("####"))
            text1 = '{} {}'.format( r['questions'], options_text) # 问题+选项
            tokenized_text1 = [self.cls_token_id] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text1)) + [self.sep_token_id]#把问题+选项从英文转换成序列
            prompt_length = len(tokenized_text1)#问题+选项 长度
            tokenized_text, total_length = tokenized_text1, len(tokenized_text1)
            if get_annotations:
                text2 = r['explanations']
                tokenized_text2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text2))
                tokenized_text = tokenized_text1 + tokenized_text2#问题+选项+解释
                tokenized_text = tokenized_text + [self.sep_token_id]
                
                if len(tokenized_text) > block_size:
                    tokenized_text = tokenized_text[:block_size]
                total_length = len(tokenized_text)
                if len(tokenized_text) < block_size:#如果长度不足block_size,就进行填充
                    tokenized_text = tokenized_text + [self.pad_token_id] * (block_size-len(tokenized_text))
            label_mrc = int(r['answers'][0]) - 1
            options_text = r['options'].split("####")

            #input_ids_mrc是opitons_input的序列表示,[5*65],5个选项都会对options_input进行编码
            #inputs_mask_mrc和inputs_segment_mrc在有词的位置为1,填充的地方为0 [5*65]
            input_ids_mrc, inputs_mask_mrc, inputs_segment_mrc = self.convert_examples_to_features(r['questions'], \
                            options_text, tokenizer, block_size, r['options_input'] if 'options_input' in r else None) # cos-e
            # input_ids_mrc = self.convert_examples_to_features(r['questions'], r['options'].split("##"), tokenizer, block_size) # medical
            if self.print_count > 0:
                print(len(tokenized_text))
                print ('example: ', text1 + text2 if get_annotations else text1)
                self.print_count = self.print_count - 1
                print("total_length: ", total_length)
            # batch, bacth_mrc, prompt_lengths, total_lengths, labels_mrc
            #tokenized_text(问题+选项+解释),input_ids_mrc(option_inputs的序列表示),inputs_mask_mrc(option_inputs的mask),inputs_segment_mrc(同上),promput_length(问题+选项的长度),
            #total_length(option_input的长度),label_mrc正确选项
            return (tokenized_text, input_ids_mrc, inputs_mask_mrc, inputs_segment_mrc, prompt_length, total_length, label_mrc)

        self.examples = self.data.apply(create_example, axis=1).to_list()
        # self.examples = self.examples[:100]

        # 先不用缓存的文件
        # print ('Saving ', len(self.examples), ' examples')
        # with open(cached_features_file, 'wb') as handle:
        #     pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def __getitem__(self, item):
        return torch.tensor(self.examples[item][0]), torch.tensor(self.examples[item][1]),  torch.tensor(self.examples[item][2]), \
            torch.tensor(self.examples[item][3]), self.examples[item][4], self.examples[item][5], self.examples[item][6]

    @staticmethod
    def convert_examples_to_features(start_ending, endings, tokenizer, max_seq_length, endings_text=None):#问题,选项, , ,背景
        #就是把ending_text编为序列

        """Loads a data file into a list of `InputBatch`s."""

        # CSQA is a multiple choice task. To perform this task using Bert,
        # we will use the formatting proposed in "Improving Language
        # Understanding by Generative Pre-Training" and suggested by
        # @jacobdevlin-google in this issue
        # https://github.com/google-research/bert/issues/38.
        #
        # Each choice will correspond to a sample on which we run the
        # inference. For a given Swag example, we will create the 4
        # following inputs:
        # - [CLS] context [SEP] choice_1 [SEP]
        # - [CLS] context [SEP] choice_2 [SEP]
        # - [CLS] context [SEP] choice_3 [SEP]
        # - [CLS] context [SEP] choice_4 [SEP]
        # - [CLS] context [SEP] choice_5 [SEP]
        # The model will output a single value for each input. To get the
        # final decision of the model, we will run a softmax over these 4
        # outputs.
        # print(start_ending)
        start_ending_tokens = tokenizer.tokenize("Q: " + start_ending) # question tokens

        choices_ids = []
        choices_mask = []
        choices_segment_ids = []
        if endings_text is not None:
            endings_text = endings_text.split("######")
            assert len(endings_text) == len(endings)
        for ending_index, ending in enumerate(endings):
            if endings_text is None:
                ending_tokens = tokenizer.tokenize("A: " + ending)
                _truncate_seq_pair(start_ending_tokens, ending_tokens, max_seq_length - 3)
                tokens = ["[CLS]"] + start_ending_tokens + ["[SEP]"] + ending_tokens + ["[SEP]"]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                segment_ids = [0] * (len(start_ending_tokens) + 2) + [1] * (len(ending_tokens) + 1)

                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                choices_ids.append(input_ids)
                choices_mask.append(input_mask)
                choices_segment_ids.append(segment_ids)
                print("not ok")
            else:
                
                ending_tokens = tokenizer.tokenize(endings_text[ending_index])
                ending_tokens = ending_tokens[:max_seq_length-2]
                tokens = ["[CLS]"] + ending_tokens + ["[SEP]"]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)#转换成序列
                input_mask = [1] * len(input_ids)
                segment_ids = [1] * (len(input_ids))

                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                choices_ids.append(input_ids)
                choices_mask.append(input_mask)
                choices_segment_ids.append(segment_ids)
        return choices_ids, choices_mask, choices_segment_ids


class GPTTSVAddMRCDataset(TSVDataset):
    def __init__(self, tokenizer, args, file_path='train', block_size=512, get_annotations=False, is_training=False):
        self.print_count = 0
        self.token_between = ", "
        self.sep_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
        self.eos_token_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)
        cached_features_file, data = self.load_data(file_path, block_size)
        self.data = data

        if get_annotations: cached_features_file = cached_features_file + '_annotated'

        # if os.path.exists(cached_features_file):
        #     print ('Loading features from', cached_features_file)
        #     with open(cached_features_file, 'rb') as handle:
        #         self.examples = pickle.load(handle)
        #     return

        # print ('Saving features from ', file_path, ' into ', cached_features_file) 

        def create_example(r):
            # text1 = '{} ? {}'.format( r['questions'], r['options']) # medical
            # cos-e :
            options_text = self.token_between.join(r['options'].split("####"))
            text1 = '{} {}'.format(r['questions'], options_text) # cos-e
            tokenized_text1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text1)) + [self.sep_token_id]
            prompt_length = len(tokenized_text1)
            tokenized_text, total_length = tokenized_text1, len(tokenized_text1)
            if get_annotations:
                text2 = r['explanations']
                tokenized_text2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text2))
                tokenized_text = tokenized_text1 + tokenized_text2
                tokenized_text = tokenized_text + [self.eos_token_id]
                total_length = len(tokenized_text)
                if len(tokenized_text) > block_size:
                    tokenized_text = tokenized_text[:block_size]
                if len(tokenized_text) < block_size:
                    tokenized_text = tokenized_text + [self.eos_token_id] * (block_size-len(tokenized_text))
            label_mrc = int(r['answers'][0]) - 1
            options_text = r['options'].split("####")

            input_ids_mrc, inputs_mask_mrc, inputs_segment_mrc = self.convert_examples_to_features(r['questions'], \
                            options_text, tokenizer, block_size, r['options_input'] if 'options_input' in r else None) # cos-e
            # input_ids_mrc = self.convert_examples_to_features(r['questions'], r['options'].split("##"), tokenizer, block_size) # medical
            if self.print_count > 0:
                print(len(tokenized_text))
                print ('example: ', text1 + text2 if get_annotations else text1)
                self.print_count = self.print_count - 1
                print("total_length: ", total_length)
            # batch, bacth_mrc, prompt_lengths, total_lengths, labels_mrc
            # 只用前两个
            return (tokenized_text, input_ids_mrc, inputs_mask_mrc, inputs_segment_mrc, prompt_length, total_length, label_mrc)

        self.examples = self.data.apply(create_example, axis=1).to_list()
        # self.examples = self.examples[:100]

        # 先不用缓存的文件
        # print ('Saving ', len(self.examples), ' examples')
        # with open(cached_features_file, 'wb') as handle:
        #     pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def __getitem__(self, item):
        return torch.tensor(self.examples[item][0]), torch.tensor(self.examples[item][1]),  torch.tensor(self.examples[item][2]), \
            torch.tensor(self.examples[item][3]), self.examples[item][4], self.examples[item][5], self.examples[item][6]

    def convert_examples_to_features(self, start_ending, endings, tokenizer, max_seq_length, endings_text=None):
        """Loads a data file into a list of `InputBatch`s."""

        # CSQA is a multiple choice task. To perform this task using Bert,
        # we will use the formatting proposed in "Improving Language
        # Understanding by Generative Pre-Training" and suggested by
        # @jacobdevlin-google in this issue
        # https://github.com/google-research/bert/issues/38.
        #
        # Each choice will correspond to a sample on which we run the
        # inference. For a given Swag example, we will create the 4
        # following inputs:
        # - [CLS] context [SEP] choice_1 [SEP]
        # - [CLS] context [SEP] choice_2 [SEP]
        # - [CLS] context [SEP] choice_3 [SEP]
        # - [CLS] context [SEP] choice_4 [SEP]
        # - [CLS] context [SEP] choice_5 [SEP]
        # The model will output a single value for each input. To get the
        # final decision of the model, we will run a softmax over these 4
        # outputs.
        # print(start_ending)
        start_ending_tokens = tokenizer.tokenize("Q: " + start_ending) # question tokens

        choices_ids = []
        choices_mask = []
        choices_segment_ids = []
        if endings_text is not None:
            endings_text = endings_text.split("######")
            assert len(endings_text) == len(endings)
        for ending_index, ending in enumerate(endings):
            if endings_text is None:
                ending_tokens = tokenizer.tokenize("A: " + ending)
                _truncate_seq_pair(start_ending_tokens, ending_tokens, max_seq_length - 3)
                tokens = start_ending_tokens + ["[SEP]"] + ending_tokens + ["[EOS]"]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                segment_ids = [0] * (len(start_ending_tokens) + 2) + [1] * (len(ending_tokens) + 1)
                choices_mask.append(len(input_ids)-1)
                padding = self.eos_token_id * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                choices_ids.append(input_ids)
                
                choices_segment_ids.append(segment_ids)
            else:
                ending_tokens = tokenizer.tokenize(endings_text[ending_index])
                ending_tokens = ending_tokens[:max_seq_length-2]
                tokens =  ending_tokens + ["[EOS]"]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                segment_ids = [1] * (len(input_ids))
                choices_mask.append(len(input_ids)-1)
                padding = [self.eos_token_id] * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                choices_ids.append(input_ids)
                
                choices_segment_ids.append(segment_ids)
        return choices_ids, choices_mask, choices_segment_ids


