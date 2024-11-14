# This code is based on "Sent-RoBERTa" implementation from https://github.com/Jihuai-wpy/SeqXGPT.
# Note: This file may contain modifications not present in the original source.

import torch
import numpy as np
import os

from tqdm import tqdm, trange
from sklearn.metrics import precision_score, recall_score
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score



class SupervisedTrainer:
    def __init__(self, data, model, en_labels, id2label, args):
        self.data = data
        self.model = model
        self.en_labels = en_labels
        self.id2label =id2label

        self.num_train_epochs = args['num_train_epochs']
        self.weight_decay = args['weight_decay']
        self.lr = args['lr']
        self.warm_up_ratio = args['warm_up_ratio']

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)

        if self.data.train_dataloader:
            self._create_optimizer_and_scheduler()

    def _create_optimizer_and_scheduler(self):
        num_training_steps = len(
            self.data.train_dataloader) * self.num_train_epochs
        no_decay = ["bias", "LayerNorm.weight"]

        named_parameters = self.model.named_parameters()
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in named_parameters
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                self.weight_decay,
            },
            {
                "params": [
                    p for n, p in named_parameters
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            betas=(0.9, 0.98),
            eps=1e-8,
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warm_up_ratio * num_training_steps,
            num_training_steps=num_training_steps)

    def train(self, threshold_f1 = None, save_dir = 'trained_model', ckpt_name='roberta_detector.pt', early_stopping = False):
        save_path = os.path.join(save_dir, ckpt_name)
        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        if threshold_f1 is None:
            sentence_result = self.test(dataloader=self.data.test_dataloader, dataloader_name='test_')
            threshold_f1 = sentence_result['macro_f1']

        for epoch in range(int(self.num_train_epochs)):
            self.model.train()
            tr_loss = 0
            nb_tr_steps = 0
            # train
            for step, inputs in enumerate(
                    tqdm(self.data.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_train_epochs}")):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                with torch.set_grad_enabled(True):
                    labels = inputs['labels']
                    output = self.model(inputs['input_ids'], inputs['masks'], inputs['labels'])
                    logits = output['logits']
                    loss = output['loss']
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    tr_loss += loss.item()
                    nb_tr_steps += 1
                    #if step % 16 == 0: 
                        #print("Running average loss:", tr_loss/nb_tr_steps)

            loss = tr_loss / nb_tr_steps
            print(f'epoch {epoch+1}: train_loss {loss}')
            # train evaluation
            #sentence_result = self.test(dataloader=self.data.train_dataloader, dataloader_name='train_')
            # test
            sentence_result = self.test(dataloader=self.data.test_dataloader, dataloader_name='test_')
            # save the model if the test performance is better, else may be overfitting
            # but continue training (may manually stop if observed that performance no longer increase)
            if early_stopping: 
                if sentence_result['macro_f1'] > threshold_f1: 
                    threshold_f1 = sentence_result['macro_f1']
                    print('*' * 120)
                    torch.save(self.model.cpu(), save_path)
                    self.model.to(self.device)
            else: 
                threshold_f1 = sentence_result['macro_f1']
                print('*' * 120)
                torch.save(self.model.cpu(), save_path)
                self.model.to(self.device)

        saved_model = torch.load(save_path)
        self.model.load_state_dict(saved_model.state_dict())
        return

    def test(self, dataloader, dataloader_name='', content_level_eval=False, convert_to_binary = None):
        self.model.eval()
        texts = []
        sentence_lengths = []
        true_labels = []
        pred_labels = []
        for step, inputs in enumerate(
                tqdm(dataloader, desc="Iteration")):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            with torch.no_grad():
                labels = inputs['labels']
                output = self.model(inputs['input_ids'], inputs['masks'], inputs['labels'])
                logits = output['logits']
                preds = output['preds']
                
                texts.extend(inputs['text'])
                sentence_lengths.extend(inputs['sentence_lengths'])
                pred_labels.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
        
        for idx, (t_label, p_label) in enumerate(zip(true_labels, pred_labels)):
            t_label = np.array(t_label)
            p_label = np.array(p_label)
            mask = t_label != -1
            t_label = t_label[mask]
            p_label = p_label[mask]
            true_labels[idx] = t_label.tolist()
            pred_labels[idx] = p_label.tolist()

        return_eval = None
        if convert_to_binary is not None: 
            self.test_en_labels = convert_to_binary

        if content_level_eval:
            # content level evaluation
            print("*" * 8, f"{dataloader_name}Content Level Evalation", "*" * 8)
            return_eval = self.content_level_eval(texts, true_labels, pred_labels, convert_to_binary)
        else:
            # sent level evalation
            print("*" * 8, f"{dataloader_name}Sentence Level Evalation", "*" * 8)
            return_eval = self.sent_level_eval(texts, sentence_lengths, true_labels, pred_labels, convert_to_binary)

        # word level evalation
        print("*" * 8, f"{dataloader_name}Word Level Evalation", "*" * 8)
        true_labels_1d = [label for t_labels in true_labels for label in t_labels]
        pred_labels_1d = [label for p_labels in pred_labels for label in p_labels]
        true_labels_1d = np.array(true_labels_1d)
        pred_labels_1d = np.array(pred_labels_1d)
        assert len(true_labels_1d) == len(pred_labels_1d), "ERROR: len(true_labels_1d) != len(pred_labels_1d)"
        accuracy = (true_labels_1d == pred_labels_1d).astype(np.float32).mean().item()
        print("Accuracy: {:.1f}".format(accuracy*100))
        
        return return_eval
    
    def predict(self, dataloader, dataloader_name='', content_level_predict=False, convert_to_binary=None): 
        self.model.eval()
        texts = []
        sentence_lengths = []
        true_labels = []
        pred_labels = []
        for step, inputs in enumerate(
                tqdm(dataloader, desc="Iteration")):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            with torch.no_grad():
                labels = inputs['labels']
                output = self.model(inputs['input_ids'], inputs['masks'], inputs['labels'])
                logits = output['logits']
                preds = output['preds']
                
                texts.extend(inputs['text'])
                sentence_lengths.extend(inputs['sentence_lengths'])
                pred_labels.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
        
        for idx, (t_label, p_label) in enumerate(zip(true_labels, pred_labels)):
            t_label = np.array(t_label)
            p_label = np.array(p_label)
            mask = t_label != -1
            t_label = t_label[mask]
            p_label = p_label[mask]
            true_labels[idx] = t_label.tolist()
            pred_labels[idx] = p_label.tolist()

        if convert_to_binary is not None: 
            self.test_en_labels = convert_to_binary

        if content_level_predict:
            # content level evaluation
            print("*" * 8, f"{dataloader_name}Content Level Prediction", "*" * 8)
            return self.content_level_predict(pred_labels, convert_to_binary)
        else:
            # sent level evalation
            print("*" * 8, f"{dataloader_name}Sentence Level Prediction", "*" * 8)
            return self.sent_level_predict(sentence_lengths, pred_labels, convert_to_binary)

    def content_level_predict(self, pred_labels, convert_to_binary): 
        pred_content_labels = []
        for pred_label in pred_labels:
            pred_common_tag = self._get_most_common_tag(pred_label, convert_to_binary)
            pred_content_labels.append(pred_common_tag[0])
        
        label_to_id = self.en_labels
        if convert_to_binary is not None: 
            label_to_id = self.test_en_labels

        pred_content_labels = [label_to_id[label] for label in pred_content_labels]
        return pred_content_labels
    
    def content_level_eval(self, texts, true_labels, pred_labels, convert_to_binary):
        true_content_labels = []
        pred_content_labels = []
        for text, true_label, pred_label in zip(texts, true_labels, pred_labels):
            true_common_tag = self._get_most_common_tag(true_label, convert_to_binary)
            true_content_labels.append(true_common_tag[0])
            pred_common_tag = self._get_most_common_tag(pred_label, convert_to_binary)
            pred_content_labels.append(pred_common_tag[0])
        
        label_to_id = self.en_labels
        if convert_to_binary is not None: 
            label_to_id = self.test_en_labels

        true_content_labels = [label_to_id[label] for label in true_content_labels]
        pred_content_labels = [label_to_id[label] for label in pred_content_labels]
        result = self._get_precision_recall_acc_macrof1(true_content_labels, pred_content_labels)
        return result
    
    def sent_level_predict(self, sentence_lengths, pred_labels, convert_to_binary):
        """
        """
        pred_sent_labels = []
        for sent_length, pred_label in zip(sentence_lengths, pred_labels):
            pred_sent_label = self.get_sent_label(sent_length, pred_label, convert_to_binary)
            pred_sent_labels.extend(pred_sent_label)
        
        label_to_id = self.en_labels
        if convert_to_binary is not None: 
            label_to_id = self.test_en_labels

        return [label_to_id[label] for label in pred_sent_labels]

    def sent_level_eval(self, texts, sentence_lengths, true_labels, pred_labels, convert_to_binary):
        """
        """
        true_sent_labels = []
        pred_sent_labels = []
        for text, sent_length, true_label, pred_label in zip(texts, sentence_lengths, true_labels, pred_labels):
            true_sent_label = self.get_sent_label(sent_length, true_label, convert_to_binary)
            pred_sent_label = self.get_sent_label(sent_length, pred_label, convert_to_binary)
            true_sent_labels.extend(true_sent_label)
            pred_sent_labels.extend(pred_sent_label)
        
        label_to_id = self.en_labels
        if convert_to_binary is not None: 
            label_to_id = self.test_en_labels

        true_sent_labels = [label_to_id[label] for label in true_sent_labels]
        pred_sent_labels = [label_to_id[label] for label in pred_sent_labels]
        result = self._get_precision_recall_acc_macrof1(true_sent_labels, pred_sent_labels)
        return result

    def get_sent_label(self, sent_lengths, label, convert_to_binary):
        sent_label = []
        offset = 0
        for sent_len in sent_lengths:
            tags = label[offset : sent_len + offset]
            offset += sent_len
            most_common_tag = self._get_most_common_tag(tags, convert_to_binary)
            sent_label.append(most_common_tag[0])
        
        if len(sent_label) == 0:
            print("empty sent label list")
        return sent_label
    
    def _get_most_common_tag(self, tags, convert_to_binary):
        """most_common_tag is a tuple: (tag, times)"""
        from collections import Counter
        
        tags = [self.id2label[tag] for tag in tags]
        tags = [tag.split('-')[-1] for tag in tags]
        if convert_to_binary: 
            tags = ['ai' if tag != 'human' else tag for tag in tags]
        tag_counts = Counter(tags)
        most_common_tag = tag_counts.most_common(1)[0]

        return most_common_tag

    def _get_precision_recall_acc_macrof1(self, true_labels, pred_labels):
        accuracy = accuracy_score(true_labels, pred_labels)
        macro_f1 = f1_score(true_labels, pred_labels, average='macro')
        print("Accuracy: {:.1f}".format(accuracy*100))
        print("Macro F1 Score: {:.1f}".format(macro_f1*100))

        precision = precision_score(true_labels, pred_labels, average=None)
        recall = recall_score(true_labels, pred_labels, average=None)
        print("Precision/Recall per class: ")
        precision_recall = ' '.join(["{:.1f}/{:.1f}".format(p*100, r*100) for p, r in zip(precision, recall)])
        print(precision_recall)

        result = {"precision":precision, "recall":recall, "accuracy":accuracy, "macro_f1":macro_f1}
        return result
