# YouTube Audio Processing App - 

## Link
To checkout this exciting webapp:

https://youtube-audio-processing-app.streamlit.app/

## Overview
The YouTube Audio Processing App is a Streamlit-based web application designed for processing YouTube audio. The app provides functionalities for downloading audio from a YouTube video, transcribing the audio, translating the transcribed text, performing Named Entity Recognition (NER) on the translated text, and summarizing the content. This README provides an overview of the app's features, installation, and usage instructions.

## Features
1. Download Audio from YouTube: Extracts and downloads only the audio from a specified YouTube video.

2. Transcribe Audio: Transcribes the downloaded audio into text.

3. Translate Text: Translates the transcribed text into English.

4. Named Entity Recognition (NER): Identifies and highlights named entities in the translated text.

5. Text Summarization: Summarizes the translated text into a concise form.

6. Styled Streamlit Elements: Customized Streamlit buttons and display elements for an enhanced user interface.

## Installation
To set up and run this application, follow these steps:

1. ### Clone the Repository:
   Clone the source code repository to your local machine.

2. ### Install Dependencies:
   The app requires several Python libraries. Install them using the following command:
   
   "pip install streamlit pytube whisper deep_translator transformers torch"

3. ### Run the App:
   Navigate to the app's directory and run it using Streamlit:

   "streamlit run app.py"

## Usage
After starting the app, follow these steps to process audio from a YouTube video:

1. Enter a YouTube URL: Input the URL of the YouTube video from which you want to process audio.

2. Download Audio: Click the 'Download Audio' button to download the audio from the provided YouTube link.

3. Transcribe Audio: Use the 'Transcribe Audio' button to transcribe the downloaded audio into text.

4. Translate Audio: Click the 'Translate Audio' button to translate the transcribed text into English.

5. Perform Named Entity Recognition: Use the 'Named Entity Recognition' button to identify and highlight entities in the translated text.

6. Summarize Text: Click the 'Summarize Text' button to generate a summary of the translated text.

7. View Results: The transcribed text, translated text, NER results, and summary will be displayed on the app's interface.

## Note
1. The app requires an active internet connection for downloading YouTube audio and using the translation and NER services.

2. Ensure that the YouTube URL is valid and the video contains clear audio for accurate transcription.
