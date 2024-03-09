import spacy
from spacy.language import Language
from spacy.matcher import Matcher
import json
from spacy.util import filter_spans

@Language.component("game_entity_matcher")
def game_entity_matcher(doc):
    matcher = Matcher(doc.vocab)
    # Define patterns
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
        [{"LOWER": "final"}, {"LOWER": "fantasy"}, {"IS_DIGIT": True}],# Example of using wildcard for numbers (e.g., Final Fantasy 7, Final Fantasy X)
        [{"LOWER": "final"}, {"LOWER": "fantasy"}, {"TEXT": {"REGEX": "^(X|V|I)+$"}}],  # Roman numerals
        [{"LOWER": "rocketleague"}, {"OP": "?"}],  # Optional single-token match
        [{"LOWER": "star"}, {"LOWER": "wars"}, {"IS_ALPHA": True, "OP": "*"}],  # Star Wars + any word
    ]
    matcher.add("GAME", patterns)
    matches = matcher(doc)
    spans = [doc[start:end] for match_id, start, end in matches]
    filtered_spans = filter_spans(spans)

    with doc.retokenize() as retokenizer:
        for span in filtered_spans:
            retokenizer.merge(span)
            # No need for span.merge() here, as retokenizer.merge(span) is correct and sufficient

    for span in filtered_spans:
        span.root.ent_type_ = "GAME"
    return doc

def predict_to_json(sentences, output_file):
    nlp = spacy.load("./ner_model")  # Ensure this path is correct
    output_data = {
        "mode": "",
        "config": {
            "hand": "",
            "mouse": "none",
            "default_events": []
        },
        "poses": {},
        "gestures": {}
    }
    split_sentences = [sentence.strip() for sentence in sentences.replace('and', ',').replace(';', ',').replace('.', ',').split(',') if sentence]

    for sentence in split_sentences:
        doc = nlp(sentence)
        gest = ""
        act = ""
        for ent in doc.ents:
            if ent.label_ == "GAME":
                output_data["mode"] = ent.text
            elif ent.label_ == "HAND":
                output_data["config"]["hand"] = ent.text
            elif ent.label_ == "GESTURE":
                gest = ent.text
            elif ent.label_ == "ACTION":
                act = ent.text
        if gest != "":
            if gest not in output_data["poses"] and act == "":
                actPlan = gesture_to_action_mapping(gest)
                output_data["poses"][gest] = {
                        "action": "key_down",
                        "args": actPlan
                    }
            elif gest not in output_data["poses"] and act:
                output_data["poses"][gest] = {
                        "action": "key_down",
                        "args": act
            }
        
    # Write the output_data dictionary to a JSON file
    with open(output_file, 'w') as json_file:
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
    predict_text = "making a kick motion to shoot the ball towards the goal"
    predict_text = "I want to play DOOM"
    predict(predict_text)
    #predict_long_text = "I want to play Fifa with my body, I want my right hand to controll movement of the player, I want to put my fist up to pass the ball, I want to sprint when I show three finger, I also want to kick the ball when I kick in real life"
    #predict_to_json(predict_long_text, "prediction_output.json")
