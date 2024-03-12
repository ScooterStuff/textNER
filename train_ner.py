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

@Language.component("gesture_entity_matcher")
def gesture_entity_matcher(doc):
    matcher = Matcher(doc.vocab)
    # Patterns based on the gestures list
    gesture_patterns = [
        [{"LOWER": "bow"}, {"LOWER": "arrow"}],
        [{"LOWER": "fighting"}, {"LOWER": "stance"}],
        [{"LOWER": "front"}, {"LOWER": "kick"}],
        [{"LOWER": "hadouken"}],
        [{"LOWER": "helicopter"}],
        [{"LOWER": "index"}, {"LOWER": "pinch"}],
        [{"LOWER": "kick"}],
        [{"LOWER": "left"}, {"LOWER": "hook"}],
        [{"LOWER": "left"}, {"LOWER": "kick"}],
        [{"LOWER": "left"}, {"LOWER": "punch"}],
        [{"LOWER": "mine"}],
        [{"LOWER": "punch"}],
        [{"LOWER": "push"}, {"LOWER": "back"}],
        [{"LOWER": "right"}, {"LOWER": "clockwise"}, {"LOWER": "circle"}],
        [{"LOWER": "right"}, {"LOWER": "hook"}],
        [{"LOWER": "right"}, {"LOWER": "kick"}],
        [{"LOWER": "right"}, {"LOWER": "punch"}],
        [{"LOWER": "uppercut"}],
        [{"LOWER": "walk"}, {"LOWER": "left"}],
        [{"LOWER": "walk"}, {"LOWER": "right"}],
    ]
    matcher
    matcher.add("GESTURE", gesture_patterns)
    matches = matcher(doc)
    spans = [doc[start:end] for match_id, start, end in matches]
    filtered_spans = filter_spans(spans)

    with doc.retokenize() as retokenizer:
        for span in filtered_spans:
            retokenizer.merge(span)

    for span in filtered_spans:
        span.root.ent_type_ = "GESTURE"
    return doc

@Language.component("pose_entity_matcher")
def pose_entity_matcher(doc):
    matcher = Matcher(doc.vocab)
    # Patterns based on the poses list
    pose_patterns = [
        [{"LOWER": "fist"}],
        [{"LOWER": "fist2"}],
        [{"LOWER": "five"}, {"LOWER": "fingers"}, {"LOWER": "pinch"}],
        [{"LOWER": "four"}, {"LOWER": "fingers"}, {"LOWER": "pinch"}],
        [{"LOWER": "full"}, {"LOWER": "pinch"}],
        [{"LOWER": "gun"}, {"LOWER": "click"}],
        [{"LOWER": "gun"}, {"LOWER": "click2"}],
        [{"LOWER": "gun"}, {"LOWER": "click3"}],
        [{"LOWER": "hand"}, {"LOWER": "backward"}],
        [{"LOWER": "hand"}, {"LOWER": "forward"}],
        [{"LOWER": "index"}, {"LOWER": "pinch"}],
        [{"LOWER": "index"}],
        [{"LOWER": "palm"}, {"LOWER": "stop"}],
        [{"LOWER": "peace"}],
        [{"LOWER": "pinky"}, {"LOWER": "2"}, {"LOWER": "up"}],
        [{"LOWER": "pinky"}, {"LOWER": "3"}, {"LOWER": "up"}],
        [{"LOWER": "pinky"}, {"LOWER": "up"}, {"LOWER": "education"}],
        [{"LOWER": "pinky"}, {"LOWER": "up"}],
        [{"LOWER": "punch"}, {"LOWER": "heavy"}],
        [{"LOWER": "punch"}, {"LOWER": "light"}],
        [{"LOWER": "shoot"}],
        [{"LOWER": "three"}, {"LOWER": "fingers"}, {"LOWER": "pinch"}, {"LOWER": "hand"}, {"LOWER": "closed"}],
        [{"LOWER": "three"}, {"LOWER": "fingers"}, {"LOWER": "pinch"}],
        [{"LOWER": "three"}, {"LOWER": "fingers"}, {"LOWER": "release"}, {"LOWER": "hand"}, {"LOWER": "closed"}],
        [{"LOWER": "three"}, {"LOWER": "fingers"}],
        [{"LOWER": "thumb"}, {"LOWER": "index"}, {"LOWER": "pinch"}, {"LOWER": "hand"}, {"LOWER": "closed"}],
        [{"LOWER": "thumb"}, {"LOWER": "index"}, {"LOWER": "pinch"}],
        [{"LOWER": "thumb"}, {"LOWER": "index"}, {"LOWER": "release"}, {"LOWER": "hand"}, {"LOWER": "closed"}],
        [{"LOWER": "thumb"}, {"LOWER": "index"}, {"LOWER": "release"}],
        [{"LOWER": "thumb"}, {"LOWER": "middle"}, {"LOWER": "pinch"}],
        [{"LOWER": "thumb"}, {"LOWER": "middle"}, {"LOWER": "release"}],
        [{"LOWER": "thumb"}, {"LOWER": "pinky"}, {"LOWER": "pinch"}],
        [{"LOWER": "thumb"}, {"LOWER": "pinky"}, {"LOWER": "release"}],
        [{"LOWER": "thumb"}, {"LOWER": "ring"}, {"LOWER": "pinch"}],
        [{"LOWER": "thumb"}, {"LOWER": "ring"}, {"LOWER": "release"}],
        [{"LOWER": "thumb"}, {"LOWER": "up"}],
    ]
    matcher.add("POSE", pose_patterns)
    matches = matcher(doc)
    spans = [doc[start:end] for match_id, start, end in matches]
    filtered_spans = filter_spans(spans)

    with doc.retokenize() as retokenizer:
        for span in filtered_spans:
            retokenizer.merge(span)

    for span in filtered_spans:
        span.root.ent_type_ = "POSE"
    return doc



def train_ner(model_dir="./ner_model", new_data=TRAIN_DATA, n_iter=200):
    # Check if model directory exists and model is loadable
    if Path(model_dir).exists():
        print(f"Loading existing model from: {model_dir}")
        nlp = spacy.load(model_dir)
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
    for matcher_component in ["game_entity_matcher", "gesture_entity_matcher", "pose_entity_matcher"]:
        if matcher_component not in nlp.pipe_names:
            nlp.add_pipe(matcher_component, last=True)
    
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