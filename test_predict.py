from predict import predict_to_json
import json
import os
from predict import gesture_to_action_mapping


def test_predict_to_json(tmp_path):
    output_file = tmp_path / "prediction_output.json"
    predict_to_json("I want to play Tetris with my left hand, Rotate item using hadouken", str(output_file))
    
    with open(output_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    assert data["mode"] == "Tetris"
    assert "hadouken" in [pose["files"] for pose in data["poses"]]


def test_gesture_to_action_mapping():
    assert gesture_to_action_mapping("fist") == "pass"
    assert gesture_to_action_mapping("three fingers") == "run"
    assert gesture_to_action_mapping("unmapped gesture") == "none"


def test_predict_to_json_identifies_gestures_and_poses(tmp_path):
    output_file = tmp_path / "prediction_output.json"
    sentence = "Use a fist to select and a pinch to move items in Minecraft."
    predict_to_json(sentence, str(output_file))
    
    with open(output_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Assuming your system can differentiate between gestures and poses
    gestures = [gesture["files"] for gesture in data["gestures"]]
    poses = [pose["files"] for pose in data["poses"]]
    
    assert "fist" in poses or "fist" in gestures  # Adjust based on your system's classification
    assert "pinch" in poses or "pinch" in gestures  # Adjust based on your system's classification


def test_action_mapping_in_output_json(tmp_path):
    output_file = tmp_path / "prediction_output.json"
    sentence = "To jump in the game, do a double tap with two fingers."
    predict_to_json(sentence, str(output_file))
    
    with open(output_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    assert any(action["action"]["tmpt"] == "jump" for action in data["gestures"] + data["poses"])


def test_json_output_structure(tmp_path):
    output_file = tmp_path / "output.json"
    sentence = "Navigate the menu using the palm gesture in Mario."
    predict_to_json(sentence, str(output_file))
    
    expected_keys = {"mode", "orientation", "landmark", "poses", "gestures"}
    
    with open(output_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        assert not expected_keys - data.keys(), "Missing or additional keys in JSON output."
