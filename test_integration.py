import unittest
import json
from pathlib import Path
from spacy.util import compile_infix_regex
from predict import predict_to_json
import spacy
import time

class TestIntegrationNERModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the NLP model, assuming it's already trained and saved
        cls.nlp = spacy.load("./ner_model")

        # If your predict.py uses a specific NLP model, ensure it's loaded here
        # cls.nlp.add_pipe("game_entity_matcher")
        # cls.nlp.add_pipe("gesture_entity_matcher")
        # cls.nlp.add_pipe("pose_entity_matcher")

    def test_full_pipeline(self):
        test_sentence = "I want to play Minecraft using hand gestures, specifically pinch to jump."
        expected_output = {
            "mode": "Minecraft",
            "poses": [],
            "gestures": [{"files": "pinch", "action": {"tmpt": "jump", "class": "", "method": "click", "args": []}}],
        }

        # Temporary output file for testing
        temp_output_file = Path("./temp_test_output.json")
        predict_to_json(test_sentence, str(temp_output_file))

        with open(temp_output_file, 'r', encoding='utf-8') as file:
            output_data = json.load(file)

        # Clean up the temporary file
        temp_output_file.unlink()

        self.assertEqual(output_data["mode"], expected_output["mode"], "The game mode does not match.")
        self.assertTrue(any(gesture["files"] == "pinch" for gesture in output_data["gestures"]), "The gesture does not match.")
        self.assertTrue(any(gesture["action"]["tmpt"] == "jump" for gesture in output_data["gestures"]), "The action does not match.")



class TestModelAccuracy(unittest.TestCase):
    def test_combined_entity_recognition(self):
        test_texts = [
            "I want to play Final Fantasy XV with my feet.",
            "Shoot using two fingers in Call of Duty.",
        ]
        expected_entities = [
            ("Final Fantasy XV", "GAME"),
            ("two fingers", "POSE"),
        ]

        for test_text, (expected_text, expected_label) in zip(test_texts, expected_entities):
            doc = TestIntegrationNERModel.nlp(test_text)
            found_entities = [(ent.text, ent.label_) for ent in doc.ents]

            self.assertIn((expected_text, expected_label), found_entities, f"Entity {expected_text} with label {expected_label} not found in text: {test_text}")

class TestComplexCommandJSONCreation(unittest.TestCase):
    def test_complex_command_json_output(self):
        test_sentence = "In FIFA, use my right hand to control the player, fist to shoot, and swipe to pass."
        expected_actions = ["shoot", "pass"]  # Simplified for demonstration

        temp_output_file = Path("./complex_command_output.json")
        predict_to_json(test_sentence, str(temp_output_file))

        with open(temp_output_file, 'r', encoding='utf-8') as file:
            output_data = json.load(file)

        temp_output_file.unlink()  # Clean up

        # Check if all expected actions are present in either gestures or poses
        actions_from_output = [action["action"]["tmpt"] for action in output_data["gestures"] + output_data["poses"]]
        for expected_action in expected_actions:
            self.assertIn(expected_action, actions_from_output, f"Expected action {expected_action} not found in output.")

class TestErrorHandling(unittest.TestCase):
    def test_unrecognized_input(self):
        test_sentence = "I want to play a game that doesn't exist using gestures that aren't defined."
        expected_response = {"error": "Unrecognized game or gesture"}

        temp_output_file = Path("./error_handling_output.json")
        predict_to_json(test_sentence, str(temp_output_file))

        with open(temp_output_file, 'r', encoding='utf-8') as file:
            output_data = json.load(file)

        temp_output_file.unlink()  # Clean up

        self.assertEqual(output_data.get("error"), expected_response["error"], "The system did not properly handle unrecognized input.")

class TestSystemLoad(unittest.TestCase):
    def test_system_performance_under_load(self):
        test_sentences = ["Play Tetris with my left hand.", "Jump in Minecraft by clapping hands.", "Shoot in Call of Duty using two fingers."] * 10  # Repeat to simulate load
        start_time = time.time()

        for sentence in test_sentences:
            temp_output_file = Path(f"./load_test_output_{time.time()}.json")
            predict_to_json(sentence, str(temp_output_file))
            temp_output_file.unlink()  # Immediate cleanup

        end_time = time.time()
        duration = end_time - start_time

        self.assertLess(duration, 10, "The system took too long to process commands under load.")

class TestExternalIntegration(unittest.TestCase):
    @mock.patch('external_library.some_function')
    def test_integration_with_external_library(self, mock_function):
        mock_function.return_value = "Expected result"
        result = your_function_that_calls_some_function()

        self.assertEqual(result, "Expected result", "The integration with the external library did not work as expected.")


if __name__ == "__main__":
    unittest.main()
