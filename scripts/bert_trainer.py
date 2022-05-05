""" ============================================================================
BERT Trainer
This is an executable script which trains a fresh spacy model for named entity
recognition, using BERT instead of SpaCy. The model's weights are saved to a
folder named 'bert_taxon_model'. The majority of the code here is from a colab
notebook called Custom Named Entity Recognition with BERT

Methods:
    train           Trains an NER model for a given number of iterations. Saves
                    the model to a folder 'bert_ner_model'
    test            Tests our pretrained 'bert_ner_model' and gives us
                    accuracy, precision, and recall scores.
    demo            Prints out some example named entities in some sentences.

Usage: Identical to the spacy_trainer python file.

The model is saved as bert_ner_model.
============================================================================ """
import copy
import os
import pandas as pd
import numpy as np
import spacy
from spacy.tokens import DocBin
from iob_converter import IOB_CONVERTER
from sklearn.metrics import accuracy_score
from seqeval.metrics import classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
from transformers import pipeline

from iob_converter import IOB_CONVERTER
from copious_loader import COPIOUS_LOADER
from random_loader import RANDOM_LOADER

# Initial Parameters
MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 8
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10
DEVICE = 'cpu'
# DEVICE = 'cuda'
LABELS_TO_IDS = {'O': 0, 'B-TAXON': 1, 'I-TAXON': 2}
IDS_TO_LABELS = {0: 'O', 1: 'B-TAXON', 2: 'I-TAXON'}


class dataset(Dataset):
    """
    This class was directly taken from:
    https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb#scrollTo=Eh3ckSO0YMZW
    """
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        # step 1: get the sentence and word labels
        sentence = self.data.sentence[index].strip().split()
        word_labels = self.data.word_labels[index].split(",")

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                            is_split_into_words=True,
                            return_offsets_mapping=True,
                            max_length=self.max_len,
                            padding='max_length',
                            truncation=True)

        # step 3: create token labels only for first word pieces of each tokenized word
        labels = [LABELS_TO_IDS[label] for label in word_labels]
        # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
        # create an empty array of -100 of length max_length
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100

        # set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                # overwrite label
                encoded_labels[idx] = labels[i]
                i += 1

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)

        return item

    def __len__(self):
        return self.len


def load_data(tokenizer, train):
    """
    A helper function that returns the data that we need. Only called once, but
    looks nice in Atom when you can use that arrow thingy to collapse the whole
    function!
    Input: distinction between train & test data output. train=True means we
    get our train data loader, train=False means we're getting test data.
    """
    if train:
        random = RANDOM_LOADER.create_dataset("./data/gene_result.csv", 'Org_name', "./data/sentences.txt", 40)
        copious = COPIOUS_LOADER.create_dataset('./data/copious_published/train')
        train_dataset = IOB_CONVERTER.build_csv(random+copious)
        training_set = dataset(train_dataset, tokenizer, MAX_LEN)
        train_params = {'batch_size': TRAIN_BATCH_SIZE,
                        'shuffle': True,
                        'num_workers': 0
                        }
        training_loader = DataLoader(training_set, **train_params)
        return training_loader
    else:
        copious = COPIOUS_LOADER.create_dataset('./data/copious_published/test')
        test_dataset = IOB_CONVERTER.build_csv(copious)
        testing_set = dataset(test_dataset, tokenizer, MAX_LEN)
        test_params = {'batch_size': VALID_BATCH_SIZE,
                        'shuffle': True,
                        'num_workers': 0
                        }
        testing_loader = DataLoader(testing_set, **test_params)
        return testing_loader


def loop(model, optimizer, training_loader, epoch):
    """
    Method to train the NER model!
    Note that this acts on the global variable model, defined above. This is
    probably not good practice. Just a heads up!
    """

    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()
    for idx, batch in enumerate(training_loader):

        ids = batch['input_ids'].to(DEVICE, dtype = torch.long)
        mask = batch['attention_mask'].to(DEVICE, dtype = torch.long)
        labels = batch['labels'].to(DEVICE, dtype = torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
        loss = outputs[0]
        tr_logits = outputs[1]
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)

        if idx % 10==0 and idx != 0:
            loss_step = tr_loss/nb_tr_steps
            print(f"Training loss per 10 training steps: {loss_step}")

        # compute training accuracy
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)

        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))

        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        tr_labels.extend(labels)
        tr_preds.extend(predictions)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=MAX_GRAD_NORM
        )

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")


def train():
    """
    Method to run through the ephochs & save our weights.
    """
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(LABELS_TO_IDS))
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    training_loader = load_data(tokenizer, True)

    for epoch in range(EPOCHS):
        print(f"Training epoch: {epoch + 1}")
        loop(model, optimizer, training_loader, epoch)

    directory = "./bert_ner_model"

    if not os.path.exists(directory):
        os.makedirs(directory)
    tokenizer.save_vocabulary(directory)
    model.save_pretrained(directory)
    print('All files saved')


def test():
    """
    Tests our model!
    """
    trained_model = BertForTokenClassification.from_pretrained('./bert_ner_model', num_labels=len(LABELS_TO_IDS))
    tokenizer = BertTokenizerFast.from_pretrained('./bert_ner_model')
    testing_loader = load_data(tokenizer, False)

    trained_model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):

            ids = batch['input_ids'].to(DEVICE, dtype = torch.long)
            mask = batch['attention_mask'].to(DEVICE, dtype = torch.long)
            labels = batch['labels'].to(DEVICE, dtype = torch.long)

            output = trained_model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = output[0]
            eval_logits = output[1]

            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)

            # compute evaluation accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, trained_model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)

            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)

            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            eval_labels.extend(labels)
            eval_preds.extend(predictions)

            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    labels = [IDS_TO_LABELS[id.item()] for id in eval_labels]
    predictions = [IDS_TO_LABELS[id.item()] for id in eval_preds]

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    true_pos = 0
    false_pos = 0
    false_neg = 0

    for i in range(len(labels)):
        if labels[i] == 'B-TAXON':
            if labels[i] == predictions[i]:
                true_pos += 1
            else:
                false_neg += 1
        else:
            if labels[i] != predictions[i]:
                false_pos += 1
    try:
        print('B-taxon precision: ', true_pos/(true_pos + false_pos))
        print('B-taxon recall: ', true_pos/(true_pos + false_neg))
    except:
        print('something happened')
    for i in range(len(labels)):
        if labels[i] == 'I-TAXON':
            if labels[i] == predictions[i]:
                true_pos += 1
            else:
                false_neg += 1
        else:
            if labels[i] != predictions[i]:
                false_pos += 1
    try:
        print('I-taxon precision: ', true_pos/(true_pos + false_pos))
        print('I-taxon recall: ', true_pos/(true_pos + false_neg))
    except:
        print('something happened')


def demo():
    text = 'in Aquaman, Arthur controls sharks, more commonly known as Squalus carcharias, with his mind.'
    trained_model = BertForTokenClassification.from_pretrained('./bert_ner_model', num_labels=len(LABELS_TO_IDS))
    tokenizer = BertTokenizerFast.from_pretrained('./bert_ner_model')
    nlp = pipeline('ner', model=trained_model, tokenizer=tokenizer)
    ner_results = nlp(text)
    for word in ner_results:
        if word['entity'] == 'LABEL_1' or word['entity'] == 'LABEL_2':
            print(word['word'])


# This is (generally) the only thing you'd have to worry about!
# train()
test()
# demo()
