import spacy
from spacy.training import Example
from spacy.util import filter_spans
import random
from pathlib import Path
from train_data import TRAIN_DATA
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt

import spacy
from spacy.training import Example
import random
import matplotlib.pyplot as plt
from pathlib import Path

# Assuming TRAIN_DATA is defined somewhere in the scope

def train_ner(model_dir, new_data, n_iter=10):
    if Path(model_dir).exists():
        nlp = spacy.load(model_dir)  # Load an existing model
    else:
        nlp = spacy.blank("en")  # Create a new blank model
        nlp.add_pipe("ner")

    # Add labels
    for _, annotations in new_data:
        for ent in annotations.get("entities"):
            nlp.get_pipe("ner").add_label(ent[2])

    # Train
    loss_values = []
    with nlp.disable_pipes(*(pipe for pipe in nlp.pipe_names if pipe != "ner")):
        optimizer = nlp.begin_training()
        for _ in range(n_iter):
            random.shuffle(new_data)
            losses = {}
            for text, annotations in new_data:
                example = Example.from_dict(nlp.make_doc(text), annotations)
                nlp.update([example], drop=0.5, losses=losses)
            loss_values.append(losses['ner'])

    return loss_values

# Function to plot losses
def plot_losses(train_data_sizes, loss_values, title):
    for data_size, losses in zip(train_data_sizes, loss_values):
        epochs = list(range(1, len(losses) + 1))
        plt.plot(epochs, losses, marker='o', linestyle='-', label=f'Training Loss with {data_size}% of data')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Paths for the different models (assuming you want to save/load these models)
model_paths = ["./ner_models/full", "./ner_models/half", "./ner_models/quarter"]

# Data subsets
full_data = TRAIN_DATA
half_data = TRAIN_DATA[:len(TRAIN_DATA)//2]
quarter_data = TRAIN_DATA[:len(TRAIN_DATA)//4]

# List to store training data sizes
train_data_sizes = [100, 50, 25]  # Representing full, half, and quarter datasets

# List to store loss values for different data sizes
all_loss_values = []

# Training with different data sizes
for model_path, data in zip(model_paths, [full_data, half_data, quarter_data]):
    print(f"Training with {len(data)} samples...")
    loss_values = train_ner(model_path, data, n_iter=100)
    all_loss_values.append(loss_values)

# Plot all the losses
plot_losses(train_data_sizes, all_loss_values, "Training Loss Comparison")
