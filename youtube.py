import streamlit as st
import pytube as pt
import whisper
from deep_translator import GoogleTranslator
from transformers import AutoModelForTokenClassification, AutoTokenizer, BartTokenizer, BartForConditionalGeneration
import torch
import random

# Function definitions for all tasks (download_youtube_audio, transcribe_audio, etc.)
def download_youtube_audio(youtube_url):
    yt = pt.YouTube(youtube_url)
    stream = yt.streams.filter(only_audio=True).first()
    filename = stream.download()
    return filename

def transcribe_audio(file_path):
    model = whisper.load("base")
    result = model.transcribe(file_path)
    return result['text']

def translate_text(text, chunk_size=1000):
    translator = GoogleTranslator(source='auto', target='en')
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    translated_text = ''
    for chunk in chunks:
        try:
            translated_chunk = translator.translate(chunk)
            translated_text += translated_chunk + ' '
        except Exception as e:
            translated_text += " [Error in translation] "
    return translated_text

# Function for Named Entity Recognition
def named_entity_recognition(text):
    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = inputs.tokens()
    
    # Generate color for each entity type
    entity_colors = {}
    for prediction in predictions[0]:
        entity = model.config.id2label[prediction.item()]
        if entity != 'O' and entity not in entity_colors:
            entity_colors[entity] = "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])

    # Associate tokens with their entities
    entities = []
    for token, prediction in zip(tokens, predictions[0]):
        entity = model.config.id2label[prediction.item()]
        if entity != 'O':
            entities.append((token, entity, entity_colors[entity]))
        else:
            entities.append((token, None, None))
    
    return entities, entity_colors

# Function to generate HTML with colored entity highlights
def render_ner_results(entities, entity_colors):
    html = ""
    for word, entity, color in entities:
        if entity:
            html += f'<span style="background-color: {color};">{word}</span> '
        else:
            html += word + " "
    html += "<br><br>"

    # Add legend for entity types
    for entity, color in entity_colors.items():
        html += f'<span style="background-color: {color};">{entity}</span> '

    return html

def summarize_text(text):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    inputs = tokenizer([text], max_length=1024, return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=500, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to create a styled button
def styled_button(title, description, key, color):
    st.markdown(f"""
        <style>
            div.stButton > button {{
            border: 2px solid {color};
            border-radius:10px;
            color: {color};
            height: 100px;
            width: 100%;
            margin: 5px 0;
            padding: 10px;
        }}
        div.stButton > button:hover {{
            border-color: {color};
            background-color: {color};
            color: white;
        }}
        </style>""", unsafe_allow_html=True)
    return st.button(f"{title}\n{description}", key=key)

# Streamlit app layout
st.title('YouTube Audio Processing App')

youtube_url = st.text_input('Enter a YouTube Link')

# Button for downloading audio
if styled_button('Download Audio', '(Only the audio from the video will be downloaded)', 'download_audio', '#007BFF'):
    if youtube_url:
        with st.spinner('Downloading audio...'):
            audio_file = download_youtube_audio(youtube_url)
            st.session_state['audio_file'] = audio_file
            st.success(f"Audio downloaded: {audio_file}")
    else:
        st.error('Please enter a YouTube URL.')

# Button for transcribing audio
if styled_button('Transcribe Audio', '(Downloaded Audio + Transcription)', 'transcribe_audio', '#6C757D'):
    if youtube_url:
        # Ensure audio is downloaded
        if 'audio_file' not in st.session_state:
            with st.spinner('Downloading audio...'):
                audio_file = download_youtube_audio(youtube_url)
                st.session_state['audio_file'] = audio_file

        with st.spinner('Transcribing audio...'):
            transcribed_text = transcribe_audio(st.session_state['audio_file'])
            st.session_state['transcribed_text'] = transcribed_text
    else:
        st.error('Please enter a YouTube URL.')

# Button for translating text
if styled_button('Translate Audio', '(Downloaded Audio + Transcription + Translation)', 'translate_text', '#28A745'):
    if youtube_url:
        # Ensure audio is downloaded and transcribed
        if 'audio_file' not in st.session_state:
            with st.spinner('Downloading audio...'):
                audio_file = download_youtube_audio(youtube_url)
                st.session_state['audio_file'] = audio_file

        if 'transcribed_text' not in st.session_state:
            with st.spinner('Transcribing audio...'):
                transcribed_text = transcribe_audio(st.session_state['audio_file'])
                st.session_state['transcribed_text'] = transcribed_text

        with st.spinner('Translating text...'):
            translated_text = translate_text(st.session_state['transcribed_text'])
            st.session_state['translated_text'] = translated_text
    else:
        st.error('Please enter a YouTube URL.')

# Button for NER
if styled_button('Named Entity Recognition', '(Downloaded Audio + Transcription + Translation + NER)', 'ner', '#FFC107'):
    if youtube_url:
        # Ensure audio is downloaded, transcribed, and translated
        if 'audio_file' not in st.session_state:
            with st.spinner('Downloading audio...'):
                audio_file = download_youtube_audio(youtube_url)
                st.session_state['audio_file'] = audio_file

        if 'transcribed_text' not in st.session_state:
            with st.spinner('Transcribing audio...'):
                transcribed_text = transcribe_audio(st.session_state['audio_file'])
                st.session_state['transcribed_text'] = transcribed_text

        if 'translated_text' not in st.session_state:
            with st.spinner('Translating text...'):
                translated_text = translate_text(st.session_state['transcribed_text'])
                st.session_state['translated_text'] = translated_text

        with st.spinner('Performing NER...'):
            entities, entity_colors = named_entity_recognition(st.session_state['translated_text'])
            ner_html = render_ner_results(entities, entity_colors)
            st.session_state['ner_results'] = ner_html
    else:
        st.error('Please enter a YouTube URL.')

# Button for Summarization
if styled_button('Summarize Text', '(Downloaded Audio + Transcription + Translation + NER + Summary)', 'summarize', '#17A2B8'):
    if youtube_url:
        # Ensure audio is downloaded, transcribed, translated, and NER performed
        if 'audio_file' not in st.session_state:
            with st.spinner('Downloading audio...'):
                audio_file = download_youtube_audio(youtube_url)
                st.session_state['audio_file'] = audio_file

        if 'transcribed_text' not in st.session_state:
            with st.spinner('Transcribing audio...'):
                transcribed_text = transcribe_audio(st.session_state['audio_file'])
                st.session_state['transcribed_text'] = transcribed_text

        if 'translated_text' not in st.session_state:
            with st.spinner('Translating text...'):
                translated_text = translate_text(st.session_state['transcribed_text'])
                st.session_state['translated_text'] = translated_text

        if 'ner_results' not in st.session_state:
            with st.spinner('Performing NER...'):
                entities, entity_colors = named_entity_recognition(st.session_state['translated_text'])
                ner_html = render_ner_results(entities, entity_colors)
                st.session_state['ner_results'] = ner_html

        with st.spinner('Summarizing text...'):
            summary = summarize_text(st.session_state['translated_text'])
            st.session_state['summary'] = summary
    else:
        st.error('Please enter a YouTube URL.')

# Displaying results
if 'transcribed_text' in st.session_state:
    st.text_area("Transcribed Text", st.session_state['transcribed_text'], height=150)

if 'translated_text' in st.session_state:
    st.text_area("Translated Text", st.session_state['translated_text'], height=150)

if 'ner_results' in st.session_state:
    st.markdown(st.session_state['ner_results'], unsafe_allow_html=True)

if 'summary' in st.session_state:
    st.text_area("Summary", st.session_state['summary'], height=150)
