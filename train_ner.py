import spacy
from spacy.training import Example
from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy.util import filter_spans
import random
from pathlib import Path
from itertools import chain
from train_data import TRAIN_DATA
from spacy.language import Language
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Function to add game matcher to an NLP object
@Language.component("game_entity_matcher")
def game_entity_matcher(doc):
    matcher = Matcher(doc.vocab)
    patterns = [
        [{"LOWER": "minecraft"}],
        [{"LOWER": "rocket"}, {"LOWER": "league"}],
        [{"LOWER": "tetris"}],
        [{"LOWER": "horizon"}],
        [{"LOWER": "fifa"}],
        [{"LOWER": "final"}, {"LOWER": "fantasy"}],
        [{"LOWER": "mario"}],
        [{"LOWER": "poker"}],
        [{"LOWER": "roblox"}],
        [{"LOWER": "final"}, {"LOWER": "fantasy"}, {"IS_DIGIT": True}],
        [{"LOWER": "final"}, {"LOWER": "fantasy"}, {"TEXT": {"REGEX": "^(X|V|I)+$"}}],
        [{"LOWER": "rocketleague"}, {"OP": "?"}],
        [{"LOWER": "star"}, {"LOWER": "wars"}, {"IS_ALPHA": True, "OP": "*"}],
    ]
    matcher.add("GAME", patterns)
    matches = matcher(doc)
    spans = [doc[start:end] for match_id, start, end in matches]
    filtered_spans = filter_spans(spans)

    with doc.retokenize() as retokenizer:
        for span in filtered_spans:
            retokenizer.merge(span)

    for span in filtered_spans:
        span.root.ent_type_ = "GAME"
    return doc

@Language.component("game_entity_matcher")
def entity_matcher_component(doc, matcher):
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]
    filtered_spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in filtered_spans:
            retokenizer.merge(span)
    for span in filtered_spans:
        span.root.ent_type_ = "GAME"
    return doc


def train_ner(model_dir="./ner_model", new_data=TRAIN_DATA, n_iter=10):
    # Check if model directory exists and model is loadable
    if Path(model_dir).exists():
        print(f"Loading existing model from: {model_dir}")
        nlp = spacy.load(model_dir)  # Load the existing model
    else:
        print("Creating a new model")
        nlp = spacy.blank("en")  # Create a blank Language class
        if "ner" not in nlp.pipe_names:
            ner = nlp.add_pipe("ner", last=True)
        else:
            ner = nlp.get_pipe("ner")
        # Add new entity labels to the NER model
        for _, annotations in new_data:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])

    # Check if the component already exists in the pipeline
    if "game_entity_matcher" not in nlp.pipe_names:
        nlp.add_pipe("game_entity_matcher", last=True)
    else:
        nlp.remove_pipe("game_entity_matcher")
        nlp.add_pipe("game_entity_matcher", last=True)
    
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # Only train NER
        if not Path(model_dir).exists():
            optimizer = nlp.begin_training()
        else:
            optimizer = nlp.resume_training()
        for itn in range(n_iter):
            random.shuffle(new_data)
            losses = {}
            for text, annotations in new_data:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.5, losses=losses, sgd=optimizer)
            print(f"Losses at iteration {itn}: {losses}")

    # Save model to output directory
    output_dir = Path(model_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print(f"Saved model to: {output_dir}")

if __name__ == "__main__":
    train_ner()