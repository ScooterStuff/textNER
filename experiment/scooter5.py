import spacy
from spacy.training import Example
from spacy.util import filter_spans
import random
from pathlib import Path
from train_data import TRAIN_DATA
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt

def train_ner(model_dir="./ner_models", new_data=TRAIN_DATA, n_iter=150):
    # Check if a model exists at the specified directory
    if Path(model_dir).exists():
        print(f"Loading existing model from: {model_dir}")
        nlp = spacy.load(model_dir)  # Load the existing model
    else:
        print("Creating a new model")
        nlp = spacy.blank("en")  # Create a blank English model if none exists
        nlp.add_pipe("ner")

    # Ensure the NER pipe is added and get it
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Add new entity labels to the NER model
    for _, annotations in new_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    
    loss_values = []  # List to store loss values per epoch

    # Get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # Only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(new_data)  # Shuffle the training data before each epoch
            losses = {}  # Reset losses dictionary for the new epoch
            for text, annotations in new_data:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.5, losses=losses)
            print(f"Losses at iteration {itn}: {losses['ner']}")
            loss_values.append(losses['ner'])  # Append the loss after the epoch

    return loss_values



def plot_loss(loss_values):
    plt.plot(loss_values, marker='o', linestyle='-', label='Training Loss per Epoch', markersize=2)  # Reduced marker size
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    loss_values = train_ner()  # Make sure this returns the list of loss values
    plot_loss(loss_values)
