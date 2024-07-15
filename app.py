import os
import instaloader
import ffmpeg
import wave
import json
import vosk
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['TEMP_FOLDER'] = 'temp/'
app.config['MODEL_FOLDER'] = 'vosk_model/'

def extract_audio(video_path, audio_path):
    stream = ffmpeg.input(video_path)
    stream = ffmpeg.output(stream, audio_path)
    ffmpeg.run(stream)

def transcribe_audio(audio_path, model_path):
    model = vosk.Model(model_path)
    wf = wave.open(audio_path, "rb")

    if wf.getnchannels() != 1:
        raise ValueError("Audio file must be mono")

    recognizer = vosk.KaldiRecognizer(model, wf.getframerate())

    transcription = ""

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            transcription += result.get('text', '') + " "

    final_result = json.loads(recognizer.FinalResult())
    transcription += final_result.get('text', '')

    return transcription

def download_instagram_media(url, download_folder):
    L = instaloader.Instaloader(download_videos=True, download_pictures=True)
    post = instaloader.Post.from_shortcode(L.context, url.split("/")[-2])
    L.download_post(post, target=download_folder)
    caption = post.caption
    media_path = next(
        (os.path.join(download_folder, file) for file in os.listdir(download_folder)
         if file.endswith(('.mp4', '.jpg', '.jpeg', '.png'))), None)
    return caption, media_path

@app.route('/process', methods=['POST'])
def process_instagram_url():
    try:
        data = request.json
        url = data['url']
        
        download_folder = os.path.join(app.config['TEMP_FOLDER'], 'download')
        os.makedirs(download_folder, exist_ok=True)
        
        caption, media_path = download_instagram_media(url, download_folder)
        
        if media_path.endswith('.mp4'):
            audio_path = os.path.splitext(media_path)[0] + '.wav'
            extract_audio(media_path, audio_path)
            transcription = transcribe_audio(audio_path, app.config['MODEL_FOLDER'])
        else:
            transcription = "No audio to transcribe."
        
        result = {
            'caption': caption,
            'transcription': transcription
        }
        
        # Placeholder for sending result to Gemini API
        gemini_response = send_to_gemini_api(result)
        
        return jsonify(gemini_response)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

def send_to_gemini_api(data):
    gemini_url = "https://api.gemini.com/factcheck"  # Replace with actual Gemini API endpoint
    response = requests.post(gemini_url, json=data)
    return response.json()

if __name__ == '__main__':
    app.run(debug=True)
