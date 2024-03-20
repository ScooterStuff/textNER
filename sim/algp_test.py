from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine, euclidean, cityblock
import numpy as np

# List of models to test
models = [
    'all-MiniLM-L6-v2',
    'bert-base-nli-mean-tokens',
    'roberta-base-nli-stsb-mean-tokens'
    # Add other models you want to test
]

# Example target and phrases (replace with your actual data)
target_phrase = "destroy"
phrases = ["break", "jump", "run", "place"]

# Function to encode phrases using a specified model
def encode_phrases(model_name, phrases):
    model = SentenceTransformer(model_name)
    return model.encode(phrases)

# Function to calculate similarities/distance
def calculate_similarity(target_embedding, phrase_embeddings, distance_type="cosine"):
    similarities = []
    for embedding in phrase_embeddings:
        if distance_type == "cosine":
            similarity = 1 - cosine(target_embedding, embedding)
        elif distance_type == "euclidean":
            similarity = 1 / (1 + euclidean(target_embedding, embedding))  # Inverted to make it a similarity measure
        elif distance_type == "manhattan":
            similarity = 1 / (1 + cityblock(target_embedding, embedding))  # Inverted to make it a similarity measure
        similarities.append(similarity)
    return similarities

# Dictionary to hold the accuracy of each model and technique
accuracy_results = {}

# Main experiment loop
for model_name in models:
    # Encode the target and phrases
    target_embedding = encode_phrases(model_name, [target_phrase])[0]
    phrase_embeddings = encode_phrases(model_name, phrases)
    
    # Calculate similarities for each technique
    for technique in ["cosine", "euclidean", "manhattan"]:
        sim_scores = calculate_similarity(target_embedding, phrase_embeddings, technique)
        
        # Find the index of the highest similarity score
        best_match_index = np.argmax(sim_scores)
        
        # Compare the best match to the known correct answer ('break' in this example)
        correct = phrases[best_match_index] == "break"
        
        # Calculate accuracy (for the purposes of this example, it's binary correct/incorrect)
        accuracy = int(correct)
        
        # Store the accuracy result
        accuracy_results[(model_name, technique)] = accuracy

# Print the accuracy results
for model_technique, acc in accuracy_results.items():
    print(f"Model: {model_technique[0]}, Technique: {model_technique[1]}, Accuracy: {acc}")
