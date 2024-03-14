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
    game = ""
    
    # Process each sentence to find entities
    # split_sentences = [sentence.strip() for sentence in sentences.replace('and', ',').replace(';', ',').replace('.', ',').split(',') if sentence]
    split_sentences = [sentence.strip() for sentence in sentences.replace(';', ',').replace('.', ',').split(',') if sentence]
    for sentence in split_sentences:
        doc = nlp(sentence)
        actions = []
        pos = ""
        ges = ""
        for ent in doc.ents:
            if ent.label_ == "GAME":
                output_data["mode"] = ent.text
                game = ent.text
            elif ent.label_ == "ORI":
                output_data["orientation"] = ent.text
            elif ent.label_ == "LANDMARK":
                output_data["landmark"] = ent.text
            elif ent.label_ == "ACTION-O":
                actions.append(ent.text)  # Collect actions for later use
            elif ent.label_ == "POSES":
                pos = ent.text
            elif ent.label_ == "GESTURE":
                ges = ent.text
        # Assuming the last action in the sentence applies to the gestures/poses found
        if actions and pos:
            entity_data = {
                    "files": ent.text,
                    "action": {
                        "tmpt": motion_to_action_mapping(actions[-1], game),  # Placeholder for action, to be filled later
                        "class": "",  # Placeholder for similarity-based class
                        "method": "hold",
                        "args": []  # Placeholder for similarity-based args
                    }
                }
            output_data["poses"].append(entity_data)
        elif actions and ges:
            entity_data = {
                    "files": ent.text,
                    "action": {
                        "tmpt": motion_to_action_mapping(actions[-1], game),  # Placeholder for action, to be filled later
                        "class": "",  # Placeholder for similarity-based class
                        "method": "click",
                        "args": []  # Placeholder for similarity-based args
                    }
                }
            output_data["poses"].append(entity_data)
        

    # Write the output_data dictionary to a JSON file
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(output_data, json_file, indent=4)

def motion_to_action_mapping(motion, game):
    games = {
        "Minecraft": ["place", "mine", "break", "inventory", "punch", "jump", "crouch", "walk", "run", "sprint"],
        "Roblox": ["jump"],
        "Tetris": ["rotate","drop", "switch"]
    }
    if game in games:
        # Use the similarties_match function to find the most similar action
        return similarties_match(motion, games[game])
    else:
        return "Game not found"
    
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import pandas as pd

# Assuming you have initialized and fine-tuned your model as per sim_nlp.py
model_path = "./sim/fine-tuned-model"  # Path to your fine-tuned model
model = SentenceTransformer(model_path)

def similarties_match(target_phrase, possible_phrases):
    # Generate embeddings for the target phrase and possible phrases
    target_embedding = model.encode(target_phrase)
    possible_phrase_embeddings = model.encode(possible_phrases)
    
    # Calculate similarities between the target phrase and each possible phrase
    similarities = {}
    for phrase, embedding in zip(possible_phrases, possible_phrase_embeddings):
        similarity = 1 - cosine(target_embedding, embedding)  # Compute similarity
        similarities[phrase] = similarity
    
    # Find the most similar phrase
    most_similar_phrase = max(similarities, key=similarities.get)
    # for phrase, similarity in similarities.items():
    #     print(f"Similarity to '{phrase}': {similarity:.4f}")
    
    return most_similar_phrase
    
def action_to_button_mapping(action, game):
    # Define a nested dictionary with mappings for each game
    game_mappings = {
        "football": {
            "fist": "pass",
            "three fingers": "run",
            "kick": "kick"
        },
        "basketball": {
            "fist": "shoot",
            "three fingers": "dribble",
            "two hands": "block"
        },
        "soccer": {
            "fist": "throw-in",
            "foot tap": "pass",
            "kick": "kick"
        }
    }
    
    # Retrieve the specific game mapping, or default to an empty dictionary if game is not found
    mapping = game_mappings.get(game, {})
    # Return the action for the given motion, default to "none" if no mapping found
    return mapping.get(action, "none")

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
    motion = "swing sword"
    game = "Minecraft"
    action = motion_to_action_mapping(motion, game)
    print(f"For the motion '{motion}' in the game '{game}', the most similar action is '{action}'.")
