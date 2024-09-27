import os
from tempfile import NamedTemporaryFile
import streamlit as st
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
from pydub import AudioSegment, effects
from pydub.silence import split_on_silence
import speech_recognition as sr
from gtts import gTTS
from deep_translator import GoogleTranslator
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import google.generativeai as genai

# Configure Gemini API
gemini_api_key = 'AIzaSyCim_x5N84uXxVwByni-Ysd0IGx_GTzIKo'
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# Configure safety for the Gemini API
safe = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Function to extract audio from video
def extract_audio_from_video(video_path):
    with NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(temp_audio_file.name)
        return temp_audio_file.name

# Function to preprocess audio
def preprocess_audio(audio_path_mp3):
    audio = AudioSegment.from_file(audio_path_mp3, format="mp3")
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio = effects.normalize(audio)
    
    with NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
        audio.export(temp_wav_file.name, format="wav")
        return temp_wav_file.name


def transcribe_audio_chunks(chunks):
    recognizer = sr.Recognizer()
    full_text = []

    for i, chunk in enumerate(chunks):
        with NamedTemporaryFile(suffix=".wav", delete=False) as chunk_file:
            chunk.export(chunk_file.name, format="wav")
            with sr.AudioFile(chunk_file.name) as source:
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data)
                except (sr.UnknownValueError, sr.RequestError):
                    text = ""
            full_text.append(text)
    
    return " ".join(full_text)

# Function to translate text using Deep Translator
def translate_text(text, dest_lang='hi'):
    translator = GoogleTranslator(source='auto', target=dest_lang)
    translated_text = translator.translate(text)
    return translated_text

# Function to summarize text using Gemini API
def shorten_text(input_text):
    response = model.generate_content(f'''
    You are an assistant that will summarize text into a shorter form,
    while keeping the enthusiasm of the original text.
    The text may be of any language, but mostly focused on Indian Languages.
    The text you have to summarize and shorten is : "{input_text}"
    ''', safety_settings=safe)

    return response.text

# Function to convert text to audio
def text_to_audio(text, lang='hi'):
    tts = gTTS(text=text, lang=lang, slow=False)
    
    with NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
        tts.save(temp_audio_file.name)
        audio = AudioSegment.from_file(temp_audio_file.name)
        with NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
            audio.export(temp_wav_file.name, format="wav")
            return temp_wav_file.name

# Function to process video chunks and replace audio
def process_video_chunks(input_video_path, translated_audio_paths):
    video = VideoFileClip(input_video_path)
    video_chunks = []
    
    chunk_duration = 60  # 60 seconds per chunk
    for start_time in range(0, int(video.duration), chunk_duration):
        end_time = min(start_time + chunk_duration, video.duration)
        chunk = video.subclip(start_time, end_time)
        
        if translated_audio_paths:
            translated_audio_path = translated_audio_paths.pop(0)
            translated_audio = AudioFileClip(translated_audio_path)
            chunk = chunk.set_audio(translated_audio)
        
        video_chunks.append(chunk)
    
    final_video = concatenate_videoclips(video_chunks)
    output_path = "output_video.mp4"
    final_video.write_videofile(output_path, codec="libx264")
    
    return output_path

# Function to handle the full video dubbing process in segments
def process_video_to_dubbed_video(input_video_path, target_language='hi'):
    # Extract and preprocess audio from video
    audio_output_path = extract_audio_from_video(input_video_path)
    preprocessed_audio_path = preprocess_audio(audio_output_path)

    # Split audio into chunks
    audio = AudioSegment.from_wav(preprocessed_audio_path)
    chunks = split_on_silence(audio, min_silence_len=700, silence_thresh=audio.dBFS-16, keep_silence=700)
    
    # Process each chunk separately for transcription, translation, and summarization
    translated_audio_paths = []
    full_transcription = []
    full_summarized_text = []
    
    for chunk in chunks:
        # Transcribe each chunk
        chunk_transcription = transcribe_audio_chunks([chunk])
        full_transcription.append(chunk_transcription)
        
        # Translate and summarize each transcription
        translated_text = translate_text(chunk_transcription, target_language)
        summarized_text = shorten_text(translated_text)
        full_summarized_text.append(summarized_text)
        
        # Generate translated audio for each summarized text chunk
        translated_audio = text_to_audio(summarized_text, lang=target_language)
        translated_audio_paths.append(translated_audio)

    # Process video chunks with new audio
    final_video_path = process_video_chunks(input_video_path, translated_audio_paths)
    
    # Combine transcriptions and summaries
    full_transcription_text = " ".join(full_transcription)
    full_summarized_text_str = " ".join(full_summarized_text)
    
    return final_video_path, full_summarized_text_str, full_transcription_text

# Streamlit Interface
st.title("Video Dubbing Application with Enhanced Translation and Summarization")

input_video_path = st.text_input("Enter the path to the input video:")
target_language = st.selectbox("Select the target language:", ["hi", "ta", "ur"])

if st.button("Process Video"):
    final_video_path, summarized_text, transcription = process_video_to_dubbed_video(input_video_path, target_language)
    
    st.success("Dubbed video processing completed successfully.")
    
    # Display video
    st.video(final_video_path)
    
    # Create a container to hold the transcription and summarized text
    with st.container():
        # Show transcription
        if st.checkbox("Show Original Transcription"):
            st.subheader("Original Transcription")
            st.text(transcription)
        
        # Show summarized text
        if st.checkbox("Show Summarized Text"):
            st.subheader("Summarized Text")
            st.text(summarized_text)
    
    # Provide download option for the video
    with open(final_video_path, "rb") as f:
        st.download_button(
            label="Download Final Video",
            data=f,
            file_name="output_video.mp4",
            mime="video/mp4"
        )