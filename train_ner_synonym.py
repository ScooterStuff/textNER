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

TRAIN_DATA2 = augment_data_with_synonyms(TRAIN_DATA)

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

def add_game_matcher(nlp):
    matcher = Matcher(nlp.vocab)
    patterns = [
        [{"LOWER": "minecraft"}],
        [{"LOWER": "rocket"}, {"LOWER": "league"}],
        [{"LOWER": "tetris"}],
        [{"LOWER": "horizon"}],
        [{"LOWER": "fifa"}],
        [{"LOWER": "final"}, {"LOWER": "fantasy"}],
        [{"LOWER": "mario"}],
        # Add more patterns for other games
    ]
    matcher.add("GAME", patterns)
    return matcher

def entity_matcher(doc, matcher):
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]  # Create Span objects
    filtered_spans = filter_spans(spans)  # Filter overlapping spans
    with doc.retokenize() as retokenizer:
        for span in filtered_spans:
            retokenizer.merge(span)
            span.merge()
    for span in filtered_spans:
        span.label_ = "GAME"
    return doc

def train_ner(model_dir="./ner_model", new_data=TRAIN_DATA, n_iter=10):
    # Check if a model exists at the specified directory
    if Path(model_dir).exists():
        print(f"Loading existing model from: {model_dir}")
        nlp = spacy.load(model_dir)  # Load the existing model
    else:
        print("Creating a new model")
        nlp = spacy.blank("en")  # Create a blank English model if none exists
        if "ner" not in nlp.pipe_names:
            ner = nlp.add_pipe("ner", last=True)
        else:
            ner = nlp.get_pipe("ner")
        # Add new entity labels to the NER model
        for _, annotations in new_data:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])

    # Continue training the model
    if "ner" in nlp.pipe_names:
        ner = nlp.get_pipe("ner")
    
    # Disable other pipeline components during training
    with nlp.disable_pipes(*[pipe for pipe in nlp.pipe_names if pipe != "ner"]):
        optimizer = nlp.resume_training()
        for itn in range(n_iter):
            losses = {}
            for text, annotations in new_data:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.5, losses=losses, sgd=optimizer)
            print(f"Losses at iteration {itn}: {losses}")

    # Save the updated model
    nlp.to_disk(model_dir)
    print(f"Saved model to: {model_dir}")

if __name__ == "__main__":
    train_ner()