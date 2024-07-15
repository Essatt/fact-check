import os
import logging
from flask import Flask, request, jsonify
import instaloader
import subprocess
import json
from vosk import Model, KaldiRecognizer

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
logging.debug("Logging is configured correctly.")

# Initialize the Instaloader instance
L = instaloader.Instaloader()

# Path to Vosk model
vosk_model_path = "vosk_model/vosk-model-small-en-us-0.15"

@app.route('/process', methods=['POST'])
def process_instagram_url():
    try:
        data = request.json
        instagram_url = data.get('url')
        shortcode = instagram_url.split("/")[-2]
        base_download_folder = "downloaded_posts"
        download_folder = os.path.join(base_download_folder, shortcode)
        logging.debug(f"Downloading Instagram media from {instagram_url} to {download_folder}")

        # Ensure download folder exists
        if not os.path.exists(download_folder):
            os.makedirs(download_folder)

        # Set the target folder to the base directory
        L.dirname_pattern = download_folder

        # Download Instagram media
        logging.debug(f"Starting Instagram media download from URL: {instagram_url}")
        post = instaloader.Post.from_shortcode(L.context, shortcode)
        L.download_post(post, target='')

        # Extract caption and media file path
        caption = None
        media_path = None
        for file in os.listdir(download_folder):
            logging.debug(f"Checking file: {file}")
            if file.endswith('.mp4'):
                media_path = os.path.join(download_folder, file)
                logging.debug(f"Found media file: {media_path}")
            if file.endswith('.txt'):
                with open(os.path.join(download_folder, file), 'r') as caption_file:
                    caption = caption_file.read()
                logging.debug(f"Post caption: {caption}")

        if media_path:
            # Extract audio from video
            audio_path = media_path.replace('.mp4', '.wav')
            logging.debug(f"Extracting audio from {media_path} to {audio_path}")
            command = [
                'ffmpeg', '-y', '-i', media_path, '-vn', '-acodec', 'pcm_s16le',
                '-ac', '1', '-ar', '44100', audio_path
            ]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logging.debug(f"ffmpeg stdout: {result.stdout.decode('utf-8')}")
            logging.debug(f"ffmpeg stderr: {result.stderr.decode('utf-8')}")
            if result.returncode != 0:
                logging.error(f"ffmpeg failed with return code {result.returncode}")
                raise Exception(f"ffmpeg error: {result.stderr.decode('utf-8')}")
            logging.debug(f"Audio extraction complete: {audio_path}")

            # Transcribe audio
            transcription_path = audio_path.replace('.wav', '-transcription.txt')
            logging.debug(f"Transcribing audio from {audio_path} using model {vosk_model_path}")
            model = Model(vosk_model_path)
            rec = KaldiRecognizer(model, 44100)

            with open(audio_path, 'rb') as audio_file:
                data = audio_file.read()
                rec.AcceptWaveform(data)

            result = rec.Result()
            transcription = json.loads(result).get('text', '')
            logging.debug(f"Transcription complete: {transcription}")

            # Save transcription
            with open(transcription_path, 'w') as transcription_file:
                transcription_file.write(transcription)
            logging.debug(f"Transcription saved to {transcription_path}")

        else:
            logging.error("No media file found in the download folder")

        return jsonify({'caption': caption, 'transcription': transcription if media_path else 'No audio to transcribe.'})
    
    except Exception as e:
        logging.error(f"Error processing Instagram URL: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
