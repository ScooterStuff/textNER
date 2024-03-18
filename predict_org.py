import spacy
import json
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from train_ner import game_entity_matcher, gesture_entity_matcher, pose_entity_matcher

# Path to your fine-tuned model and NER model
MODEL_PATH = "./fine-tuned-model"
NER_MODEL_PATH = "./ner_model"

# Load models outside of functions to avoid reloading them on each function call
try:
    # Load models
    sentence_model = SentenceTransformer(MODEL_PATH)
    nlp = spacy.load(NER_MODEL_PATH)
except Exception as e:
    print(f"Error loading models: {e}")
    # Exit or handle accordingly
    exit(1)

def similarties_match(target_phrase, possible_phrases):
    """
    Create embbed for target and possible phrases then calculate the max similarities between target and possible phrases.
    """
    try:
        # Generate embeddings for the target phrase and possible phrases
        target_embedding = sentence_model.encode(target_phrase)
        possible_phrase_embeddings = sentence_model.encode(possible_phrases)
        
        similarities = {}
        for phrase, embedding in zip(possible_phrases, possible_phrase_embeddings):
            similarity = 1 - cosine(target_embedding, embedding)  # Compute similarity
            similarities[phrase] = similarity
        
        most_similar_phrase, highest_similarity = max(similarities.items(), key=lambda item: item[1], default=(None, 0))
        # for phrase, similarity in similarities.items():
        #     print(f"Similarity to '{phrase}': {similarity:.4f}")
        
        if highest_similarity < 0.2:
            return "none"
        else:
            return most_similar_phrase
    except Exception as e:
        print(f"Error in similarity matching: {e}")
        return "none"
    
def motion_to_action_mapping(motion, game):
    """
    Use similarities match to map the closest motion by user to in game action.
    """
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

def predict_to_json(sentences, output_file):
    try:
        split_sentences = preprocess_sentences(sentences)
        output_data = initialize_output_structure()
        predict(split_sentences, output_data)

        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(output_data, json_file, indent=4)
    except Exception as e:
        print(f"Error during prediction or file writing: {e}")
        # Handle accordingly, e.g., try again, log error, etc.


def preprocess_sentences(sentences):
    """
    Splits and preprocesses sentences for further processing.
    """
    return [sentence.strip() for sentence in sentences.replace(';', ',').replace('.', ',').split(',') if sentence]

def initialize_output_structure():
    """
    Initializes the structure of the output data.
    """
    return {"mode": "", "orientation": "", "landmark": "", "poses": [], "gestures": []}

def predict(split_sentences, output_data):
    """
    Processes entities identified in the document and updates the output data accordingly.
    """
    gestures = ['bow_arrow', 'fighting_stance', 'front_kick', 'hadouken', 'helicopter', 'index_pinch', 'kick', 'left_hook', 'left_kick', 'left_punch', 'mine', 'punch', 'push_back', 'right_clockwise_circle', 'right_hook', 'right_kick', 'right_punch', 'uppercut', 'walk_left', 'walk_right']
    poses = ['fist', 'fist2', 'five_fingers_pinch', 'four_fingers_pinch', 'full_pinch', 'gun_click', 'gun_click2', 'gun_click3', 'hand_backward', 'hand_forward', 'index_pinch', 'index', 'palm_stop', 'peace', 'pinky_2_up', 'pinky_3_up', 'pinky_up_education', 'pinky_up', 'punch_heavy', 'punch_light', 'shoot', 'three_fingers_pinch_hand_closed', 'three_fingers_pinch', 'three_fingers_release_hand_closed', 'three_fingers', 'thumb_index_pinch_hand_closed', 'thumb_index_pinch', 'thumb_index_release_hand_closed', 'thumb_index_release', 'thumb_middle_pinch', 'thumb_middle_release', 'thumb_pinky_pinch', 'thumb_pinky_release', 'thumb_ring_pinch', 'thumb_ring_release', 'thumb_up']  
    #Should add pinch
    game = ""
    
    for sentence in split_sentences:
        doc = nlp(sentence)
        pos = ""
        ges = ""
        
        actions = []
        for ent in doc.ents:
            label = ent.label_
            text = ent.text
            print(label, text)
            if label == "GAME":
                output_data["mode"] = text
                game = text
            elif label == "ORI":
                output_data["orientation"] = text
            elif label == "LANDMARK":
                output_data["landmark"] = text
            elif ent.label_ == "ACTION-O":
                    actions.append(ent.text)
            elif ent.label_ == "POSES":
                pos = text
            elif ent.label_ == "GESTURE":
                ges = text
        if actions and pos:
            entity_data = {
                    "files": similarties_match(pos, poses),
                    "action": {
                        "tmpt": actions[-1],  # Placeholder for action, to be filled later
                        "class": motion_to_action_mapping(actions[-1], game),  # Placeholder for similarity-based class
                        "method": "hold",
                        "args": []  # Placeholder for similarity-based args
                    }
                }
            output_data["poses"].append(entity_data)
        elif actions and ges:
            entity_data = {
                    "files":  similarties_match(ges, gestures),
                    "action": {
                        "tmpt": actions[-1],  # Placeholder for action, to be filled later
                        "class": motion_to_action_mapping(actions[-1], game),  # Placeholder for similarity-based class
                        "method": "click",
                        "args": []  # Placeholder for similarity-based args
                        }
                    }
            output_data["gestures"].append(entity_data)


def main():
    predict_text = "I want to play Tetris with my left hand, Rotate item using hadouken, To crouch just do a thumb down, Use the index pinch to interact with objects"
    # predict_text = "I want to play Minecraft. I want to jump using three fingers"
    predict_text = "I want to play Minecraft with my right arm, I want to jump when I pose thumb down, I want to do index pinch to place down a block, three fingers to destroy."
    predict_to_json(predict_text, "prediction_output_two.json")

if __name__ == "__main__":
    main()
