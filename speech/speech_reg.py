from pydub import AudioSegment
import speech_recognition as sr

# Convert MP3 to WAV
def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    audio = AudioSegment.from_mp3(mp3_file_path)
    audio.export(wav_file_path, format="")

# Transcribe audio file to text
def transcribe_audio(wav_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_file_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text

wav_file = "c:/Users/ASUS/Desktop/TEXTNER-1/speech/audio.wav"  # Corrected path

transcribed_text = transcribe_audio(wav_file)
print(transcribed_text)


#I want to play FIFA with my body I want my right hand to control movement of the player I want to put my fist up to pass the ball I want to sprint when I show three finger I also want to kick the ball when I Kick In Real Life
#I want to play Fifa with my body, I want my right hand to controll movement of the player, I want to put my fist up to pass the ball, I want to sprint when I show three finger, I also want to kick the ball when I kick in real life