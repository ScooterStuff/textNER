import spacy
from spacy.language import Language
from spacy.matcher import Matcher
import json
from spacy.util import filter_spans

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

def predict_to_json(sentences, output_file):
    nlp = spacy.load("./ner_model")  # Ensure this path is correct
    output_data = {
        "mode": "",
        "orientation": "",
        "landmark": "",
        "poses": [],
        "gestures": []
    }
    
    # Process each sentence to find entities
    # split_sentences = [sentence.strip() for sentence in sentences.replace('and', ',').replace(';', ',').replace('.', ',').split(',') if sentence]
    split_sentences = [sentence.strip() for sentence in sentences.replace(';', ',').replace('.', ',').split(',') if sentence]
    for sentence in split_sentences:
        doc = nlp(sentence)
        actions = []
        for ent in doc.ents:
            if ent.label_ == "GAME":
                output_data["mode"] = ent.text
            elif ent.label_ == "ORI":
                output_data["orientation"] = ent.text
            elif ent.label_ == "LANDMARK":
                output_data["landmark"] = ent.text
            elif ent.label_ == "ACTION-O":
                actions.append(ent.text)  # Collect actions for later use
            elif ent.label_ in ["POSES", "GESTURE"]:
                # Prepare the dictionary structure for poses or gestures
                entity_data = {
                    "files": ent.text,
                    "action": {
                        "tmpt": "",  # Placeholder for action, to be filled later
                        "class": "",  # Placeholder for similarity-based class
                        "method": "click" if ent.label_ == "GESTURE" else "hold",
                        "args": []  # Placeholder for similarity-based args
                    }
                }
                # Decide where to place the entity based on its label
                if ent.label_ == "POSES":
                    output_data["poses"].append(entity_data)
                elif ent.label_ == "GESTURE":
                    output_data["gestures"].append(entity_data)
        
        # Assuming the last action in the sentence applies to the gestures/poses found
        if actions:
            last_action = actions[-1]
            for entity_list in [output_data["poses"], output_data["gestures"]]:
                for entity in entity_list:
                    entity["action"]["tmpt"] = last_action

    # Write the output_data dictionary to a JSON file
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(output_data, json_file, indent=4)

def gesture_to_action_mapping(gesture):
    mapping = {
        #This is automated version
        "fist": "pass",
        "three fingers": "run",
        "kick": "kick"
    }
    return mapping.get(gesture, "none")  # Default to "none" if no mapping found

def predict(text):
    nlp = spacy.load("./ner_model")  # Ensure this path is correct
    doc = nlp(text)
    for token in doc:
        print(token.text, token.ent_type_)
    for ent in doc.ents:
        print("Entity:", ent.text, ent.label_)

if __name__ == "__main__":
    predict_text = "Rotate item using hadouken"
    predict(predict_text)
    # predict_long_text = "I want to play Fifa with my body, I want my right hand to controll movement of the player, I want to put my fist up to pass the ball, I want to sprint when I show three finger, I also want to kick the ball when I kick in real life"
    predict_long_text = "I want to play Tetris with my left hand, Rotate item using hadouken, To crouch just do a thumb down, Use the index pinch to interact with objects"
    predict_to_json(predict_long_text, "prediction_output.json")
