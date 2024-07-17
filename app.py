import os
import logging
from flask import Flask, request, jsonify
import instaloader
import subprocess
import json
from vosk import Model, KaldiRecognizer
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Set up Instagram downloader
L = instaloader.Instaloader(download_videos=True, download_comments=False)

# Configure the Gemini API client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("No GEMINI_API_KEY found. Please set the GEMINI_API_KEY environment variable.")
genai.configure(api_key=GEMINI_API_KEY)
# Choose a model
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

@app.route('/process', methods=['POST'])
def process_instagram_url():
    data = request.json
    url = data['url']
    shortcode = url.split('/')[-2]

    try:
        # Create a directory to store the downloaded media
        download_dir = f'downloaded_posts/{shortcode}'
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        
        # Download Instagram media
        logging.debug(f"Downloading Instagram media from {url}")
        post = instaloader.Post.from_shortcode(L.context, shortcode)
        L.download_post(post, target=download_dir)

        # Extract audio and save as WAV
        video_file = None
        for file in os.listdir(download_dir):
            if file.endswith('.mp4'):
                video_file = os.path.join(download_dir, file)
                break

        if video_file:
            audio_file = os.path.join(download_dir, f"{shortcode}.wav")
            logging.debug(f"Extracting audio from {video_file} to {audio_file}")
            subprocess.run(['ffmpeg', '-i', video_file, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1', audio_file])

            # Transcribe audio using Vosk
            model = Model("vosk_model/vosk-model-small-en-us-0.15")
            rec = KaldiRecognizer(model, 44100)
            transcription = ""
            with open(audio_file, 'rb') as f:
                while True:
                    data = f.read(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        result = rec.Result()
                        transcription += json.loads(result)['text'] + " "
                transcription += json.loads(rec.FinalResult())['text']
            logging.debug(f"Transcription: {transcription}")

            # Read post caption
            caption = post.caption

            # Prepare text for Gemini
            text_to_check = f"Post Caption: {caption}\n\nPost Video Transcription: {transcription}"
            logging.debug(f"Text to check: {text_to_check}")

            # Check facts with Gemini
            prompt = (
                "Fact-check the following claim from a social media post, using the below given caption and audio transcription of the post:\n\n"
                f"{text_to_check}\n\n"
                "Please assess the factual accuracy of the following social media post and any accompanying audio transcription:\n\n"
                "1. Given the following text and audio transcript from a social media post, determine if the claims made are supported by evidence. Provide a clear verdict (True, False, Partially True, or Insufficient Evidence) and a concise explanation.\n"
                "2. Find and list studies that support this claim. If you can't find any return 'No such study found'. Use only trusted sources and studies. Heavily use Google Scholar and any other scholar databases. For the studies you find list the following for each study:\n"
                "    - Links to the studies,\n"
                "    - information about the researches,\n"
                "    - who funded it\n"
                "    - does funding look sketchy? Does the company or people who funded this have any stake in the study? Do they benefit from the result in any way?\n"
                "    - which institution did the study\n"
                "    - was the study’s aim the same as the hypothesis or was it a random side fact?\n"
                "    - What was the sample size\n"
                "    - What was the sample demographic\n"
                "    - Were there any errors in sampling (not diverse enough, too small) or analysis?\n"
                "3. Find and list studies that directly disprove the claim. If you can't find any return 'No such study found'. Use only trusted sources and studies. Heavily use Google Scholar and any other scholar databases. For the studies you find list the following for each study:\n"
                "    - Links to the studies,\n"
                "    - information about the researchers,\n"
                "    - who funded it\n"
                "    - does funding look sketchy? Does the company or people who funded this have any stake in the study? Do they benefit from the result in any way?\n"
                "    - which institution did the study\n"
                "    - was the study’s aim the same as the hypothesis or was it a random side fact?\n"
                "    - What was the sample size\n"
                "    - What was the sample demographic\n"
                "    - Were there any errors in sampling (not diverse enough, too small) or analysis?"
            )

            response = gemini_model.generate_content(prompt)
            logging.debug(f"Gemini results: {response.text}")

            return jsonify({
                'caption': caption,
                'transcription': transcription,
                'gemini_results': response.text
            })

        else:
            return jsonify({'error': 'No media file found in the download folder'})

    except Exception as e:
        logging.error(f"Error processing Instagram URL: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
