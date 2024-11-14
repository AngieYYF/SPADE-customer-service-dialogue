# This code is based on "Sent-RoBERTa" implementation from https://github.com/Jihuai-wpy/SeqXGPT.
# Note: This file may contain modifications not present in the original source.

import numpy as np
import os
import random
import torch
import pandas as pd
import pickle

from tqdm import tqdm, trange
from pathlib import Path
from datasets import Dataset, DatasetDict
from torch.utils.data.dataloader import DataLoader, RandomSampler, SequentialSampler
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

def construct_bmes_labels(labels):
    prefix = ['B-', 'M-', 'E-', 'S-']
    id2label = {}
    counter = 0

    for label, id in labels.items():
        for pre in prefix:
            id2label[counter] = pre + label
            counter += 1
    
    return id2label

en_labels = {'gpt' : 0,
             'llama': 1,
             'human' : 2}

id2label = construct_bmes_labels(en_labels)

en_binary_labels = {'human' : 0, 'ai' : 1}

id2label_binary = construct_bmes_labels(en_binary_labels)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class DataManager:
    def __init__(self, dataset_df: pd.DataFrame, id2label, test_ratio, tokenizer, dataset_type, batch_size, label_pad_idx=-1, load_from_cache=False, truncate=False, train_ids=None):
        '''
            dataset_df: a dataframe containing instances of all labels
            test_ratio: the proportion of test data
            dataset_type: the type of dataset (e.g. 'next_response', 'paraphrase')
        '''
        self.random_state = 0
        set_seed(self.random_state)
        self.batch_size = batch_size # one dialogue at a time
        self.human_label = 'human'
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.tokenizer = tokenizer
        self.max_len = tokenizer.model_max_length
        self.label_pad_idx = label_pad_idx
        self.dataset_type = dataset_type
        self.truncate = truncate

        # prepare training and testing dataset
        train_df = None
        test_df = None
        if not load_from_cache: 
            # confirms format of dataset_df
            required_columns = ['label', 'dia', 'dia_no']
            for column in required_columns:
                assert column in dataset_df.columns, f"DataFrame is missing required column: {column}"
            # split into train and test dataset
            train_df, test_df = DataManager._train_test_split(dataset_df, test_ratio, self.random_state, train_ids)

        # convert to dataset
        data = dict()
        data["train"] = self.initialize_dataset("train", train_df, dataset_type, load_from_cache)
        data["test"] = self.initialize_dataset("test", test_df, dataset_type, load_from_cache)
        datasets = DatasetDict(data)

        # prepare training and testing dataloader
        self.train_dataloader = self.get_train_dataloader(datasets["train"])
        self.test_dataloader = self.get_eval_dataloader(datasets["test"])
        
    def _train_test_split(dataset_df, test_ratio, random_state, train_ids): 
        if train_ids is None: 
            unique_dia_nos = list(range(623)) #dataset_df['dia_no'].unique()
            train_ids, test_ids = train_test_split(unique_dia_nos, test_size=test_ratio, random_state=random_state)
        train_df = dataset_df[dataset_df['dia_no'].isin(train_ids)]
        test_df = dataset_df[~dataset_df['dia_no'].isin(train_ids)]
        return train_df, test_df
    
    def initialize_dataset(self, role, dataset_df, dataset_type, load_from_cache, save_dir='roberta_dataset'):
        max_len_role_ids = [1,0] * 20 # assume maximum 20 user-system turns
        
        processed_data_filename = f"{role}_{dataset_type}.pkl"
        processed_data_path = os.path.join(save_dir, processed_data_filename)

        if os.path.exists(processed_data_path) and load_from_cache:
            log_info = '*'*4 + 'Load From {}'.format(processed_data_path) + '*'*4
            print('*' * len(log_info))
            print(log_info)
            print('*' * len(log_info))
            with open(processed_data_path, 'rb') as f:
                samples_dict = pickle.load(f)
            return Dataset.from_dict(samples_dict)

        # convert dataframe into a dictionary of format {col_name: [data]}
        samples = dataset_df.to_dict(orient='list')
        samples_dict = {'input_ids': [], 'labels': [], 'text': [], 'sentence_lengths':[], 'masks':[]}

        for i in trange(dataset_df.shape[0]):
            text = samples['dia'][i]
            
            # break dialogue into utterances
            utterances = text.split("\n")
            utterances_update = []
            # ignore any sentences without any text
            for idx,utterance in enumerate(utterances):
                if utterance.strip().lower() == 'system:' or utterance.strip().lower() == 'user:':
                    continue
                utterances_update.append(utterance)
            utterances = utterances_update
            # ignore first system utterance "How may I help you", and last system response
            if utterances[0].startswith("system:"):
                utterances = utterances[1:]
            if utterances[-1].startswith("system:"): 
                utterances = utterances[:-1]
            if "next_response" in dataset_type: # for last response dataset, only look at a maximum sliding window of 4 utterances
                utterances = utterances[-4:]
            text = "\n".join(utterances)

            initial_role_index = utterances[0].startswith("system:")
            is_user = max_len_role_ids[initial_role_index:len(utterances)+initial_role_index]
            label = samples['label'][i]

            # extract input ids and labels from the dialogue
            aligned_ids, aligned_labels, sent_lengths = self.get_input_labels(utterances, is_user, label, dataset_type)
            
            # split dialogue into multiple samples if exceeded maximum length
            def split_list(lst, max_len):
                if self.truncate: 
                    return [lst[:max_len]]
                else: 
                    return [lst[i:i+max_len] for i in range(0, len(lst), max_len)]
            if self.truncate: 
                sent_lengths = [sent_lengths[0]]

            input_ids_list = split_list(aligned_ids, self.max_len)
            labels_list = split_list(aligned_labels, self.max_len)
            assert len(input_ids_list) == len(labels_list) == len(sent_lengths), f"unmatched len(input_ids_list) == len(labels_list) == len(sent_lengths)\n{text}\n{sent_lengths}\n{input_ids_list}\n{labels_list}"

            masks_list = [[1] * self.max_len for _ in range(len(input_ids_list))]

            last_list_len = len(input_ids_list[-1])
            input_ids_list[-1] += [self.tokenizer.pad_token_id] * (self.max_len - last_list_len)
            labels_list[-1] += [self.label_pad_idx] * (self.max_len - last_list_len)
            masks_list[-1] = [1] * last_list_len + [0] * (self.max_len - last_list_len)
            texts = [text for _ in range(len(input_ids_list))]

            # append processed data
            samples_dict['input_ids'].extend(input_ids_list)
            samples_dict['masks'].extend(masks_list)
            samples_dict['labels'].extend(labels_list)
            samples_dict['text'].extend(texts)
            samples_dict['sentence_lengths'].extend(sent_lengths)
            
        
        # save the processed dataset
        dir_name = os.path.dirname(processed_data_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(processed_data_path, 'wb') as f:
            pickle.dump(samples_dict, f)

        return Dataset.from_dict(samples_dict)
        
    def get_train_dataloader(self, dataset):
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          sampler=RandomSampler(dataset),
                          collate_fn=self.data_collator)

    def get_eval_dataloader(self, dataset):
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          sampler=SequentialSampler(dataset),
                          collate_fn=self.data_collator)
    
    def data_collator(self, samples):
        # samples: {'input_ids': [], 'labels': [], 'text': [], 'sentence_lengths': []}
        batch = {}

        input_ids = [sample['input_ids'] for sample in samples]
        labels = [sample['labels'] for sample in samples]
        text = [sample['text'] for sample in samples]
        sentence_lengths = [sample['sentence_lengths'] for sample in samples]
        masks = [sample['masks'] for sample in samples]

        batch['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
        batch['labels'] = torch.tensor(labels, dtype=torch.long)
        batch['masks'] = torch.tensor(masks, dtype=torch.long) 
        batch['text'] = text
        batch['sentence_lengths'] = sentence_lengths

        return batch


    def _split_en_sentence(self, sentence):
        import re
        pattern = re.compile(r'\S+') # original has \s for whitespaces as well?
        words = pattern.findall(sentence)
        return words
    
    def sequence_labels_to_ids(self, seq_len, label):
        if label == self.label_pad_idx: 
            return [label] * seq_len
        
        prefix = ['B-', 'M-', 'E-', 'S-']
        if seq_len <= 0:
            return []
        elif seq_len == 1:
            label = 'S-' + label
            return [self.label2id[label]]
        else:
            ids = []
            ids.append(self.label2id['B-'+label])
            ids.extend([self.label2id['M-'+label]] * (seq_len - 2))
            ids.append(self.label2id['E-'+label])
            return ids


    def get_input_labels(self, sentences, is_user, label, dataset_type):
        # cut the "user: " and "system: "
        start_index = [len("system: "), len("user: ")]
        tokenizer = self.tokenizer

        # break the sentences into words 
        input_ids = []
        all_labels = []
        sent_lengths = [[]]
        remaining_length = self.max_len
        for idx, (cur_is_user, sentence) in enumerate(zip(is_user, sentences)):
            # determine the sentence label, default to padding (for system)
            sent_label = self.label_pad_idx 
            if cur_is_user: 
                if "next_response" not in dataset_type or idx == len(is_user)-1: 
                    sent_label = label
            # obtain token ids
            sent_len = 0
            words = self._split_en_sentence(sentence[start_index[cur_is_user]:])
            for word in words:
                sub_tokens = tokenizer.tokenize(word)
                sent_len += len(sub_tokens)
                input_ids.extend(tokenizer.convert_tokens_to_ids(sub_tokens))
            
            if sent_len == 0: 
                continue
            
            # obtain sentence lengths and labels
            if remaining_length >= sent_len: 
                if sent_label != self.label_pad_idx: 
                    sent_lengths[-1].append(sent_len)
                remaining_length -= sent_len
            else: 
                if sent_label != self.label_pad_idx: 
                    if remaining_length != 0: 
                        sent_lengths[-1].append(remaining_length)
                    sent_lengths.append([sent_len - remaining_length])
                    remaining_length = self.max_len - sent_lengths[-1][0]
                    assert remaining_length >= 0, "single utterance exceeds the max len"
                else: 
                    remaining_length = self.max_len
                    sent_lengths.append([])
            all_labels.extend(self.sequence_labels_to_ids(sent_len, sent_label))
            
        assert len(input_ids) == len(all_labels), "len(input_ids) != len(all_labels), something error."
        
        return input_ids, all_labels, sent_lengths