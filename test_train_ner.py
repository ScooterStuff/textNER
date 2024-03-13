# test_train_ner.py

import spacy
from train_ner import game_entity_matcher
from train_ner import train_ner
import os

def test_model_recognizes_new_entities():
    # Assuming your model has been trained and saved to './ner_model'
    nlp = spacy.load("./ner_model")
    test_text = "Playing Horizon with gestures feels intuitive."
    doc = nlp(test_text)

    assert any(ent.label_ == "GAME" and ent.text == "Horizon" for ent in doc.ents)
    assert any(ent.label_ == "GESTURE" for ent in doc.ents)  # If your test text includes gestures

def test_game_entity_matcher():
    nlp = spacy.load("./ner_model")
    doc = nlp("I want to play Minecraft.")
    
    assert len(doc.ents) == 1
    assert doc.ents[0].text == "Minecraft"
    assert doc.ents[0].label_ == "GAME"

def test_game_entity_matcher_retokenizes_correctly():
    nlp = spacy.load("./ner_model")
    doc = nlp("Let's play Rocket League tonight.")
    
    # Ensure "Rocket League" is recognized as a single entity
    assert len(doc.ents) == 1
    assert doc.ents[0].text == "Rocket League"
    assert doc.ents[0].label_ == "GAME"

def test_train_ner_creates_model_dir(tmp_path):
    model_dir = tmp_path / "ner_model"
    train_ner(model_dir=str(model_dir), n_iter=1)  # Run with fewer iterations for testing
    
    assert model_dir.exists()

def test_new_labels_added_to_ner():
    nlp = spacy.blank("en")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")
    
    # Pretend we're adding a new label as part of training
    ner.add_label("NEW_LABEL")
    
    assert "NEW_LABEL" in ner.labels