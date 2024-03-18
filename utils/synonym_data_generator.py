import spacy
from spacy.training import Example
from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy.util import filter_spans
from train_data import TRAIN_DATA
import random
from pathlib import Path
from itertools import chain
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')

def synonym_replacement(text, entity_spans):
    nltk.download('wordnet', quiet=True)
    
    words = text.split()
    new_words = words.copy()
    changes = {}  # Track changes in word lengths
    
    for i, word in enumerate(words):
        synonyms = set(chain.from_iterable([syn.lemma_names() for syn in wordnet.synsets(word)]))
        synonyms.discard(word.lower())  # Discard the original word to avoid redundancy
        if synonyms:
            synonym = random.choice(list(synonyms))
            changes[i] = (len(word), len(synonym))  # Original and new length
            new_words[i] = synonym
    
    new_text = ' '.join(new_words)
    new_spans = adjust_entity_spans(text, new_text, entity_spans)
    
    return new_text, new_spans
def augment_data_with_synonyms(TRAIN_DATA):
    augmented_data = []
    for text, annotations in TRAIN_DATA:
        entity_spans = [(start, end, label) for start, end, label in annotations['entities']]
        augmented_text, new_entity_spans = synonym_replacement(text, entity_spans)
        new_annotations = {"entities": new_entity_spans}
        augmented_data.append((augmented_text, new_annotations))
    return augmented_data + TRAIN_DATA
def adjust_entity_spans(original_text, new_text, original_spans):
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, original_text, new_text)
    new_spans = []
    
    for opcode in matcher.get_opcodes():
        tag, i1, i2, j1, j2 = opcode
        if tag == 'equal':
            shift = j1 - i1
            for start, end, label in original_spans:
                if start >= i1 and end <= i2:
                    new_start = start + shift
                    new_end = end + shift
                    new_spans.append((new_start, new_end, label))
    return new_spans
TRAIN_DATA2 = augment_data_with_synonyms(TRAIN_DATA)


def create_syn_train_data(data):
    syn_data = augment_data_with_synonyms(data)
    print(syn_data)

if __name__ == "__main__":
    create_syn_train_data(TRAIN_DATA)