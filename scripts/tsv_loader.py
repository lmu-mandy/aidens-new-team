""" ============================================================================
TSV Loader
This is a class file which loads data from the corpus-species databank and
turns it into data that is right for SpaCy to use - meaning data of the form
(text, {entities: [ (start,end,label) ]})

In the final product, this file is NOT USED because the 'taxonomy' data is
not as reliable as the copious dataset.

Usage: none.
============================================================================ """
import spacy
from spacy.lang.en import English
from spacy.training import Example
from spacy.pipeline import EntityRuler
from spacy.util import minibatch, compounding

import pandas as pd
import re
import random
import warnings
from os import listdir

# from spacy_code import *

class TSV_LOADER:

    def concat_txt_file(path):
        """
        Concatonates a text file into one string.
            IN:     string file path
            OUT:    string
        """
        text = ''
        with open(path, encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                text += line
        return text

    def create_dataset(path):
        """
        Creates a datapoint from the 'tags.tsv' file.
        """
        df = pd.read_csv(path, sep='\t')

        data = []
        previous_doc = df['document'][0]
        ents = []
        for idx, row in df.iterrows():
            doc = row['document']

            if doc != previous_doc:
                text_path = './data/corpus-species/txt/' + previous_doc + '.txt'
                text = TSV_LOADER.concat_txt_file(text_path)
                data.append( (text, {'entities':ents}) )
                ents = []

            ents.append( (row['start'], row['end'], 'TAXON') )
            previous_doc = doc
        return data
