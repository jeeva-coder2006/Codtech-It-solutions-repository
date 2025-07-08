
import speech_recognition as sr
import ffmpeg

input_file="sample_audio.m4a"
output_file="converted.wav"

ffmpeg.input(input_file).output(output_file,ac=1,ar=16000).run()

# Initialize recognizer
r = sr.Recognizer()

# Load an audio file (wav format recommended)
with sr.AudioFile("converted.wav") as source:
    audio = r.record(source)

# Convert speech to text
try:
    text = r.recognize_google(audio)
    print("Transcription:", text)
except sr.UnknownValueError:
    print("Could not understand audio.")
except sr.RequestError as e:
    print("Could not request results; {0}".format(e))