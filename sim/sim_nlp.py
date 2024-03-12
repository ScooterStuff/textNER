# Import necessary libraries
from transformers import AutoModel, AutoTokenizer
import torch
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import pandas as pd
from sentence_transformers import evaluation

# Simulated dataset: Normally, you'd load a dataset from a file
data = [
    {"sentence1": "move the ball", "sentence2": "pass", "similarity": 0.9},
    {"sentence1": "move the ball", "sentence2": "walk", "similarity": 0.1},
    {"sentence1": "move the ball", "sentence2": "punt", "similarity": 0.8},
    {"sentence1": "move the ball", "sentence2": "run", "similarity": 0.2},
]

# Convert the simulated dataset to a pandas DataFrame
df = pd.DataFrame(data)

# Convert the DataFrame to a list of InputExample objects
examples = [InputExample(texts=[row['sentence1'], row['sentence2']], label=row['similarity']) for index, row in df.iterrows()]

# Initialize DataLoader
train_dataloader = DataLoader(examples, shuffle=True, batch_size=2)

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the training method using cosine similarity loss
train_loss = losses.CosineSimilarityLoss(model=model)

# Assuming a very small dataset, let's skip validation for this example.
# In a real scenario, you should split your data and use a validation set.

# Fine-tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=4,  # Adjust epochs based on your dataset size and complexity
          warmup_steps=100,
          output_path="./fine-tuned-model"  # Change this to where you want to save your model
         )

# Load the fine-tuned model (optional if you continue using the same model object)
model = SentenceTransformer("./fine-tuned-model")

# Now you can use the model as before to generate embeddings and calculate similarities
# The target phrase and phrases to compare
target_phrase = "stab"
phrases = ["walk", "pass", "punt", "run"]
phrases_button = ["A","X","Y","B"]

# Generate embeddings for each phrase
target_embedding = model.encode(target_phrase)
phrase_embeddings = model.encode(phrases)

# Calculate and print the cosine similarity between the target and each phrase
similarities = {}
for phrase, embedding in zip(phrases, phrase_embeddings):
    # Compute cosine similarity (note: 1 - cosine distance to get similarity)
    similarity = 1 - cosine(target_embedding, embedding)
    similarities[phrase] = similarity

# Find the most similar phrase
most_similar_phrase = max(similarities, key=similarities.get)
print(f"The phrase most similar to '{target_phrase}' is: '{most_similar_phrase}' with a similarity score of {similarities[most_similar_phrase]:.4f}")

# Optional: print all similarities for comparison
for phrase, similarity in similarities.items():
    print(f"Similarity to '{phrase}': {similarity:.4f}")
