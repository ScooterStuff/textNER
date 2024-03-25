from motion_game_mapper import MotionGameMapper

import speech_recognition as sr

# Transcribe audio file to text
def transcribe_audio(wav_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_file_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text

def main():
    wav_file = "C:/Users/ASUS/Desktop/textNER-1/speech/audio.wav"
    mapper = MotionGameMapper()
    predict_text = transcribe_audio(wav_file)
    mapper.predict_to_json(predict_text, "prediction_output.json")



if __name__ == "__main__":
    main()