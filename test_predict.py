import unittest
from unittest.mock import patch, MagicMock
from unittest.mock import mock_open
# Assuming the class-based implementation is saved in a file named `motion_game_mapper.py`
from motion_game_mapper import MotionGameMapper

class TestMotionGameMapper(unittest.TestCase):
    def setUp(self):
        # Mock the SentenceTransformer and Spacy model loading
        with patch('motion_game_mapper.SentenceTransformer'), patch('motion_game_mapper.spacy.load'):
            self.mapper = MotionGameMapper()
            
    @patch('json.dump')
    @patch('builtins.open', new_callable=MagicMock)
    def test_predict_to_json(self, mock_open, mock_json_dump):
        sentences = "I want to play Minecraft with my left hand, jump using two fingers."
        output_file = "test_output.json"
        self.mapper.predict_to_json(sentences, output_file)
        mock_open.assert_called_with(output_file, 'w', encoding='utf-8')
        self.assertTrue(mock_json_dump.called)

    def test_initialize_output_structure(self):
        expected = self.mapper.initialize_output_structure()
        self.assertEqual(expected, {"mode": "", "orientation": "", "landmark": "", "poses": [], "gestures": []})

    @patch('motion_game_mapper.spacy.load', MagicMock())
    @patch('motion_game_mapper.SentenceTransformer', MagicMock())
    def test_predict_without_comma(self):
        mock_game_ent = MagicMock()
        mock_game_ent.label_ = 'GAME'
        mock_game_ent.text = 'Tetris'

        mock_action_ent = MagicMock()
        mock_action_ent.label_ = 'ACTION-O'
        mock_action_ent.text = 'Rotate item'

        mock_doc = MagicMock()
        mock_doc.ents = [mock_game_ent, mock_action_ent]
        mock_nlp = MagicMock(return_value=mock_doc)

        with patch.object(self.mapper, 'nlp', new=mock_nlp):
            output_data = self.mapper.initialize_output_structure()
            self.mapper._predict_without_comma("I want to play Tetris", output_data)

        self.assertEqual(output_data["mode"], 'Tetris')

    def test_motion_to_action_mapping_returns_none_for_unknown_game(self):
        motion = "fly"
        game = "UnknownGame"
        expected = "Game not found"
        result = self.mapper.motion_to_action_mapping(motion, game)
        self.assertEqual(result, expected)

    @patch('json.dump')
    @patch('builtins.open', new_callable=MagicMock)
    def test_predict_to_json_creates_correct_file_structure(self, mock_open, mock_json_dump):
        sentences = "I want to play Minecraft; Jump using two fingers."
        output_file = "test_output_structure.json"
        self.mapper.predict_to_json(sentences, output_file)
        args, kwargs = mock_json_dump.call_args
        output_data = args[0]
        expected_keys = ["mode", "orientation", "landmark", "poses", "gestures"]
        self.assertTrue(all(key in output_data for key in expected_keys))

    def test_similarities_match_with_empty_possible_phrases(self):
        target_phrase = "mine"
        possible_phrases = []
        expected = "none"
        result = self.mapper._similarities_match(target_phrase, possible_phrases, self.mapper.sentence_model)
        self.assertEqual(result, expected)

    def test_motion_to_action_mapping_with_invalid_game_type(self):
        motion = "jump"
        game = {"name": "Minecraft"}
        with self.assertRaises(TypeError):
            self.mapper.motion_to_action_mapping(motion, game)

    @patch('json.dump')
    @patch('builtins.open', new_callable=MagicMock)
    def test_predict_to_json_with_empty_sentences(self, mock_open, mock_json_dump):
        sentences = ""
        output_file = "empty_sentences_output.json"
        self.mapper.predict_to_json(sentences, output_file)
        args, kwargs = mock_json_dump.call_args
        output_data = args[0]
        expected_keys = ["mode", "orientation", "landmark", "poses", "gestures"]
        self.assertTrue(all(key in output_data for key in expected_keys), "All expected keys should be present in the output.")
        self.assertTrue(all(not output_data[key] for key in expected_keys), "Output for empty sentences should not populate any data.")

    def test_similarities_match_with_empty_inputs(self):
        target_phrase = "mine"
        possible_phrases = []
        expected = "none"
        result = self.mapper._similarities_match(target_phrase, possible_phrases, self.mapper.sentence_model)
        self.assertEqual(result, expected, "Expected 'none' for an empty list of possible phrases.")
    
    @patch('json.dump')
    @patch('builtins.open', mock_open(), create=True)
    def test_predict_to_json_with_empty_sentences(self, mock_json_dump):
        sentences = ""
        output_file = "empty_sentences_output.json"
        self.mapper.predict_to_json(sentences, output_file)
        
        args, kwargs = mock_json_dump.call_args
        output_data = args[0]  # First argument to json.dump
        
        expected_keys = ["mode", "orientation", "landmark", "poses", "gestures"]
        # Ensure all expected keys are present and have no data populated
        self.assertTrue(all(key in output_data for key in expected_keys), "All expected keys should be present in the output.")
        self.assertTrue(all(not output_data[key] for key in expected_keys), "Output for empty sentences should not populate any data.")



if __name__ == '__main__':
    unittest.main()
