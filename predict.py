import spacy
from spacy.language import Language
from spacy.matcher import Matcher
import json
from spacy.util import filter_spans
from train_ner import game_entity_matcher
from train_ner import gesture_entity_matcher
from train_ner import pose_entity_matcher


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
