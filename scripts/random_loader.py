""" ============================================================================
Random Loader
This is a class file which loads data from the corpus-species databank and
turns it into data that is right for SpaCy to use - meaning data of the form
(text, {entities: [ (start,end,label) ]})

The data is built from 15 handwritten sentences, as well as a database of over
900 unique taxonomies. The RANDOM_LOADER class builds dataframes by randomly
selecting taxonomies and filling in <TAXON> tags in the 15 handwritten
sentences.

Usage: none on its own.
============================================================================ """

import spacy
from spacy.lang.en import English
from spacy.training import Example
from spacy.pipeline import EntityRuler

import pandas as pd
import numpy as np
import random
import re

class RANDOM_LOADER:

    def load_taxons(file_name, taxon_col_name):
        """
        Fetches taxonomy names from data
        INPUTS: File_name (name of a file), taxon_col_name (name of taxonomy column)
        OUTPUTS: list of taxonomy names.
        """

        data = pd.read_csv(file_name)
        return data[taxon_col_name].to_list()


    def load_sentences(file_name):
        """
        Retrievs dummy example sentences from a text file.
        """
        lines = []
        with open(file_name) as f:
            lines = f.readlines()
        stripped = []
        for line in lines:
            stripped.append(line.strip())
        return stripped


    def remove_duplicates(taxons):
        """
        Clears a list of duplicate elements.
        """

        dummy_dict = {}
        for name in taxons:
            if name not in dummy_dict:
                dummy_dict[name] = 0
        return [item[0] for item in dummy_dict.items()]


    def get_better_taxons(taxons):
        """
        Abbreviates the first word of the taxons.
        """

        new_taxons = []
        for taxon in taxons:
            new_taxon = taxon.split()
            if len(new_taxon) == 2:
                new_taxon[0] = str(new_taxon[0][0]) + "."
                new_taxons.append(" ".join(new_taxon))
        return new_taxons


    def create_data_point(text, bag):
        """
        Takes in a sentence and outputs: (text, {"entities": [(start, end, label)]})
        Output uses random selections from a bag of names.
        """

        sent = text
        sent_len = len(sent)
        token = "<TAXON>"
        label = "TAXON"
        entities = []

        match = re.search(token, sent)
        while bool(match) == True:
            word  = random.choice(bag)
            start = match.span()[0]
            end   = match.span()[1] - len(token) + len(word)

            entities.append( (start, end, label) )
            sent = re.sub(token, word, sent, count=1)
            match = re.search(token, sent)

        return ( sent, {"entities": entities} )


    def create_dataset(taxon_file, col_name, sentence_file, size):
        """
        Creates a list of training data of the form:
        [ (text, {"entities": [(start, end, label)]}) ]
        """

        taxons = RANDOM_LOADER.remove_duplicates( RANDOM_LOADER.load_taxons(taxon_file, col_name) )
        taxons.extend( RANDOM_LOADER.get_better_taxons(taxons) )
        sentences = RANDOM_LOADER.load_sentences(sentence_file)
        train_data = []

        for i in range(size):
            sent = random.choice(sentences)
            train_data.append( RANDOM_LOADER.create_data_point(sent, taxons) )

        return(train_data)
