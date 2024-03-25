import speech_recognition as sr

# Transcribe audio file to text
def transcribe_audio(wav_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_file_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text

wav_file = "C:/Users/ASUS/Desktop/textNER-1/speech/audio.wav"  # Corrected path

transcribed_text = transcribe_audio(wav_file)
print(transcribed_text)

