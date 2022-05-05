""" ============================================================================
Copious Loader
This is a class file which loads custom data from hand-written sentences, along
with hand-picked taxonomy names from the NCBI dataset, and converts them into
SpaCy-friendly data, AKA data of the form:
(text, {entities: [ (start,end,label) ]})

This data is made from dozens of txt files containing excerpts from scientific
papers, all dense in taxonomic names. Each txt file has an associate ann file
which informs us the indices of the taxonomic mentions.

Usage: none on its own.
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

class COPIOUS_LOADER:
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


    def create_ann_df(path):
        """
        Creates a dictionary out of an annotation file, retrieving start & end
        indices, as well as the taxon name itself, in columns.
            IN:     string file path
            OUT:    pandas dataframe
        """
        data = {'start': [], 'end': [], 'name': []}
        with open(path, encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                data_list = re.split(' |\t|\n', line)
                if data_list[1] == 'Taxon':
                    data['start'].append(int(data_list[2]))
                    data['end'].append(int(data_list[3]))
                    data['name'].append(" ".join(data_list[4:]))
        return data


    def make_spacy_datapoint(txt_path, ann_path):
        """
        Creates a datapoint that is spacy-digestable
            IN:     string path files
            OUT:    tuple of the form (text, {entities: [ (start,end,label) ]})
        """
        text = COPIOUS_LOADER.concat_txt_file(txt_path)
        dict = COPIOUS_LOADER.create_ann_df(ann_path)
        ents = []
        for i in range(len( dict['start'] )):
            new_start = dict['start'][i]
            new_end = dict['end'][i]
            ent = (new_start, new_end, 'TAXON')
            # This block makes sure there are no duplicate spans.
            is_duplicate = False
            for n in range(0,len(ents)):
                old_start = ents[n][0]
                old_end = ents[n][1]

                if new_start == old_start or new_end == old_end:
                    is_duplicate = True
                if new_start < old_start and new_end > old_start:
                    is_duplicate = True
                if new_start > old_start and new_start < old_end:
                    is_duplicate = True

                # Replace duplicate with shorter token
                if is_duplicate:
                    old_range = old_end - old_start
                    new_range = new_end - new_start
                    if new_range < old_range:
                        ents[n] = ent
                    break
            if not is_duplicate:
                ents.append(ent)
        # Finished
        return (text, {'entities':ents})


    def create_dataset(path):
        """
        Creates a dataset!
        """
        nlp = spacy.blank('en')
        files = listdir(path)
        N = len(files)
        data = []
        for i in range(0, N-1, 2):
            txt_path = path + '/' + files[i+1]
            ann_path = path + '/' + files[i]
            datapoint = ()
            try:
                datapoint = COPIOUS_LOADER.make_spacy_datapoint(txt_path, ann_path)
            except:
                print('removed 1 datapoint')
            if len(datapoint) != 0:
                with warnings.catch_warnings(record=True) as w:
                    nlp = spacy.blank('en')
                    doc = nlp.make_doc(datapoint[0])
                    example = Example.from_dict(doc, datapoint[1])
                    if len(w) != 1:
                        data.append(datapoint)
        data = COPIOUS_LOADER.trim_entity_spans(data)
        return data


    def trim_entity_spans(data: list) -> list:
        """Removes leading and trailing white spaces from entity spans.
        Args:
            data (list): The data to be cleaned in spaCy JSON format.
        Returns:
            list: The cleaned data.
        """
        invalid_span_tokens = re.compile(r'\s')

        cleaned_data = []
        for text, annotations in data:
            entities = annotations['entities']
            valid_entities = []
            for start, end, label in entities:
                valid_start = start
                valid_end = end
                while valid_start < len(text) and invalid_span_tokens.match(
                        text[valid_start]):
                    valid_start += 1
                while valid_end > 1 and invalid_span_tokens.match(
                        text[valid_end - 1]):
                    valid_end -= 1
                valid_entities.append([valid_start, valid_end, label])
            cleaned_data.append([text, {'entities': valid_entities}])
        return cleaned_data
