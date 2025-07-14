from pytubefix import YouTube
from pydub import AudioSegment
import os
import re

# Set ffmpeg path
AudioSegment.converter = r"C:\Users\shraa\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
AudioSegment.ffprobe   = r"C:\Users\shraa\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffprobe.exe"

def sanitize_filename(title):
    # Replace spaces with underscores and remove invalid characters
    return re.sub(r'[\\/*?:"<>|]', "", title).replace(" ", "_")

def download_full_audio(youtube_url, output_folder="D:/WSAI/data1/songs"):
    yt = YouTube(youtube_url)
    title = sanitize_filename(yt.title)
    output_wav = os.path.join(output_folder, f"{title}.wav")

    print(f"Downloading: {yt.title}")
    audio_stream = yt.streams.filter(only_audio=True).first()
    downloaded_file = audio_stream.download(filename="temp_audio.mp4")

    print("Converting to WAV...")
    audio = AudioSegment.from_file(downloaded_file)
    os.makedirs(output_folder, exist_ok=True)
    audio.export(output_wav, format="wav")
    print(f"Saved: {output_wav}")

    os.remove(downloaded_file)

# --- USAGE ---
youtube_url = "https://www.youtube.com/watch?v=f4dPmq6M57s"
download_full_audio(youtube_url)
