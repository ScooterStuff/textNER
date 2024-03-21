import spacy
import json
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from train_ner import game_entity_matcher, gesture_entity_matcher, pose_entity_matcher
from game_controls import games_actions, game_key_mappings
from available_gesture_and_pose import available_gestures, available_poses

# Path to your fine-tuned model and NER model
MODEL_PATH = "./fine-tuned-model"
NER_MODEL_PATH = "./ner_model"

# Try to load models outside of functions to avoid reloading them on each function call
try:
    sentence_model = SentenceTransformer(MODEL_PATH)
    nlp = spacy.load(NER_MODEL_PATH)
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

def similarties_match(target_phrase, possible_phrases):
    """
    Create embbed for target and possible phrases then calculate the max similarities between target and possible phrases.
    """
    try:
        # Generate embeddings for the target phrase and possible phrases
        target_embedding = sentence_model.encode(target_phrase)
        possible_phrase_embeddings = sentence_model.encode(possible_phrases)
        
        similarities = {
            phrase: 1 - cosine(target_embedding, embedding) 
            for phrase, embedding in zip(possible_phrases, possible_phrase_embeddings)
        }
        
        most_similar_phrase, highest_similarity = max(similarities.items(), key=lambda item: item[1], default=(None, 0))
        # for phrase, similarity in similarities.items():
        #     print(f"Similarity to '{phrase}': {similarity:.4f}")
        return "none" if highest_similarity < 0.2 else most_similar_phrase
    
    except Exception as e:
        print(f"Error in similarity matching: {e}")
        return "none"
    
def motion_to_action_mapping(motion, game):
    """
    Use similarities match to map the closest motion by user to in game action.
    """
    if game in games_actions:
        # Use the similarties_match function to find the most similar action
        return similarties_match(motion, games_actions[game])
    else:
        return "Game not found"
    
def action_to_key_input(action, game):
    """
    Maps a game action to the corresponding keyboard input.
    """
    return game_key_mappings.get(game, {}).get(action, "Action not found")

def initialize_output_structure():
    """
    Initializes the data structure for output JSON.
    """
    return {
        "mode": "",
        "orientation": "",
        "landmark": "",
        "poses": [],
        "gestures": []
    }

def predict_to_json(sentences, output_file):
    try:
        # split_sentences = preprocess_sentences(sentences)
        output_data = initialize_output_structure()
        predict_without_comma(sentences, output_data)

        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(output_data, json_file, indent=4)
    except Exception as e:
        print(f"Error during prediction or file writing: {e}")
        # Handle accordingly, e.g., try again, log error, etc.


def predict_without_comma(sentences, output_data):
    """
    Processes a single sentence to identify game-related actions and updates the output data structure with these actions.
    This function does not require actions to be separated by commas, and processes Named Entity Recognition (NER) results directly.

    Parameters:
    - sentences: A string containing the input sentence(s) to process.
    - output_data: A dictionary where the processed information will be stored.
    """
    gestures = available_gestures
    poses = available_poses

    game = ""
    doc = nlp(sentences)
    actions = []
    # Loop through the entities identified by the custom NER model.
    for ent in doc.ents:
        if ent.label_ == "GAME":
            output_data["mode"] = ent.text
            game = ent.text
        elif ent.label_ == "ORI":
            output_data["orientation"] = ent.text
        elif ent.label_ == "LANDMARK":
            output_data["landmark"] = ent.text
        else:
            actions.append((ent.label_[0], ent.text))
    
    # Pair adjacent actions for further processing. Assumes actions come in meaningful pairs.
    result = []
    for i in range(0, len(actions) - 1, 2):
        result.append((actions[i], actions[i+1]))
    
    # Process each action pair to update the output_data with the action details.
    for action1, action2 in result:
        if action1[0] == "A" or action2[0] == "A":
            pose_or_gesture, action = (action2, action1) if action1[0] == "A" else (action1, action2)
            action_type = "poses" if pose_or_gesture[0] == "P" else "gestures"
            files = similarties_match(pose_or_gesture[1], poses if pose_or_gesture[0] == "P" else gestures)
            ignaction = motion_to_action_mapping(action[1], game)
            ignkey = action_to_key_input(ignaction, game)
            action_data = {
                "files": files,
                "action": {
                    "tmpt": action[1],
                    "class": ignaction,
                    "method": "hold" if pose_or_gesture[0] == "P" else "click",
                    "args": [ignkey]
                }
            }
            output_data[action_type].append(action_data)


def main():
    predict_text = "I want to play Tetris with my left hand, Rotate item using hadouken, To crouch just do a thumb down, Use the index pinch to interact with objects"
    predict_text = "I want to play Minecraft with my right arm I want to jump when I pose thumb down I want to do index pinch to place down a block three fingers to destroy."
    predict_to_json(predict_text, "prediction_output.json")
    

if __name__ == "__main__":
    main()
