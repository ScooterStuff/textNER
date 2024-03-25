#"I want to play Minecraft with my right arm I want to jump when I pose thumb down I want to do index pinch to place down a block three fingers to destroy."
import argparse
from motion_game_mapper import MotionGameMapper

def main(predict_text):
    mapper = MotionGameMapper()
    # Use the passed predict_text instead of a hardcoded string
    mapper.predict_to_json(predict_text, "prediction_output.json")

if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Map motion game controls from natural language input.")
    # Add the positional argument for the input text
    parser.add_argument('text', type=str, help="The input text to process.")
    
    args = parser.parse_args()
    main(args.text)
