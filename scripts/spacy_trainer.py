""" ============================================================================
SpaCy Trainer
This is an executable script which trains a fresh spacy model for named entity
recognition. The model is saved to the parent directory in a folder called
taxon_ner_model. The saved model can be run elsewhere. There is also a built-in
function here for evaluation.

Methods:
    train_spacy     Trains an NER model for a given number of iterations. Saves
                    the model to a folder 'taxon_ner_model'
    test            Tests our pretrained 'taxon_ner_model' and gives us
                    accuracy, precision, and recall scores.
    demo            Prints out some example named entities in some sentences.

Usage:
The train_spacy method should be run if someone wants to train a fresh
NER model for themselves, to get a feel for the training process.

The test method should be used if someone wants to test the preexisting model
(or one they have just trained) to see the precision, recall, etc.

The demo method can be edited to tag any sentence someone wants to see tagged.
Simply edit the string called 'text' to be any sentence!

The model is saved as 'taxon_ner_model'.
============================================================================ """
import spacy
from spacy.lang.en import English
from spacy.training import Example
from spacy.pipeline import EntityRuler
from spacy.util import minibatch, compounding
from spacy.scorer import Scorer


import pandas as pd
import re
import random
import warnings
from os import listdir

from copious_loader import COPIOUS_LOADER
from tsv_loader import TSV_LOADER
from random_loader import RANDOM_LOADER


def train(iterations):
    """
    Defines our spacy loop and spits out a freshly trained model!
    """
    # nlp = spacy.blank("en")
    nlp = spacy.load('en_core_web_sm')
    if "ner" not in nlp.pipe_names:
        nlp.add_pipe("ner", last=True)

    ner = nlp.get_pipe("ner")
    ner.add_label("TAXON")

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    cop_data = COPIOUS_LOADER.create_dataset('./data/copious_published/train')
    tsv_data = TSV_LOADER.create_dataset('./data/corpus-species/filtered_tags.tsv')
    rdm_data = RANDOM_LOADER.create_dataset("./data/gene_result.csv", 'Org_name', "./data/sentences.txt", 85)

    with nlp.disable_pipes(*other_pipes):
        # optimizer = nlp.begin_training()
        for itn in range(iterations):
            rdm_data = RANDOM_LOADER.create_dataset("./data/gene_result.csv", 'Org_name', "./data/sentences.txt", 85)
            TRAIN_DATA = cop_data + rdm_data
            random.shuffle(TRAIN_DATA)

            print("Starting iteration " + str(itn))
            losses = {}
            i = 0

            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)
                nlp.update(
                    examples,
                    drop=0.5,
                    # sgd=optimizer,
                    losses=losses
                )
            print(losses)
        return nlp

def test():
    # nlp = train_spacy(40)
    # nlp.to_disk("taxon_ner_model")

    nlp = spacy.load("taxon_ner_model")

    true_pos = 0
    false_pos = 0
    false_neg = 0

    test_data = COPIOUS_LOADER.create_dataset('./data/copious_published/test')

    for text, annotations in test_data:
        pred_ents = [ent.text for ent in nlp(text).ents]
        actual_ents = [ text[ent[0]:ent[1]] for ent in annotations['entities'] ]

        for ent in pred_ents:
            if ent in actual_ents:
                true_pos += 1
            else:
                false_pos += 1
        for ent in actual_ents:
            if ent not in pred_ents:
                false_neg += 1

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f_score = 2 * (precision*recall)/(precision+recall)
    print(precision)
    print(recall)
    print(f_score)

    # Current best is 0.76, 0.83, 0.80
    # 483 / 85


def demo():
    nlp = spacy.load("taxon_ner_model")
    text = "in Aquaman, Arthur controls sharks, more commonly known as Squalus carcharias, with his mind."
    pred_ents = [ent.text for ent in nlp(text).ents]
    print(pred_ents)

    text = "in Aquaman, Arthur meets Aiden Meyer, a math student."
    pred_ents = [ent.text for ent in nlp(text).ents]
    print(pred_ents)

# train(40)
# test()
demo()
