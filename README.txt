================================================================================
===                                 TAXONICE                                 ===
===                              by Aiden Meyer                              ===
================================================================================

Welcome to TaxoNICE! TaxoNICE is designed to be a replacement for the now
defunct github repo TaxoNERD. It's job: to provide two methods of taxonomic
named entity recognition. It can be tested on random sentences to see if they
include animal names, or more useful things like scientific papers to see how
densely populated they are with taxonomic mentions.

This project includes two models that implement this taxonomic NER: one built
on the SpaCy module, the other built via fine-tuning a BERT model. As it stands
right now, the SpaCy model outperforms the BERT model by an astounding amount.

REQUIREMENTS:
numpy
pandas
spacy
sqlearn
seqeval
torch
transformers

USAGE:
The two places to look for first-time viewers is the 'demo' method of both the
bert_trainer file and the spacy_trainer file. The 'demo' methods of both have
a variable called 'text' which can be edited freely into any sentence,
preferably one with a taxonomic name included. To see a custom sentence be
tagged, assign the variable 'text' to a string of the sentence, then comment
out both the 'train' and 'test' methods at the bottom of the file, and uncomment
'demo'. When either file is ran, the output should be the entities of the input
sentence.
    NOTE: THE BERT PRE-TRAINED MODEL IS NOT IN THE GITHUB REPOSITORY. Since the
    pre-trained file is too large to fit in github, the 'test' method for BERT
    (usage detailed below) must be run. On CPU, it shouldn't take more than half
    an hour to train on eight epochs.

To re-train the (already trained & saved) models, each file comes with a 'train'
method. Simply comment out 'test' and 'demo' at the bottom of the files and
uncomment 'train', and give the file a run to get training a brand new NER
model. This will take some time. Hyperparameters such as the learning rate,
number of epochs, etc can all be adjusted.

To check if the accuracy & validation, repreat this process with the 'test'
model. The model will be tested on 20% of the Copious dataset, which can be
found inside the data folder. More information about this dataset is detailed
in the header of the copious_loader script.
