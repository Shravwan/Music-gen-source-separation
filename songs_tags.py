from pytubefix import YouTube
from pydub import AudioSegment
import os
import re

def download_audio(youtube_url, output_path="audio_full.mp4"):
    yt = YouTube(youtube_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_stream.download(filename=output_path)
    return output_path, yt.length, yt.description

def convert_to_wav(input_file, output_file):
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format="wav")

def timestamp_to_seconds(ts):
    parts = list(map(int, ts.split(":")))
    return sum(x * 60**i for i, x in enumerate(reversed(parts)))

def extract_chapters(description, duration):
    lines = description.splitlines()
    timestamps = []

    for line in lines:
        line = line.strip()

        # Format 1: Timestamp at the start (e.g., 00:00:54 Song Title)
        match1 = re.match(r"^(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+)", line)
        if match1:
            timestamp = timestamp_to_seconds(match1.group(1))
            tag = match1.group(2).strip()
            timestamps.append((timestamp, tag))
            continue

        # Format 2: Title first, timestamp at the end (e.g., Title : 00:00:54)
        match2 = re.match(r"^(?:\d+\.\s*)?(.+?)\s*[:：]\s*(\d{1,2}:\d{2}(?::\d{2})?)$", line)
        if match2:
            tag = match2.group(1).strip()
            timestamp = timestamp_to_seconds(match2.group(2))
            timestamps.append((timestamp, tag))

    tags = []
    for i, (start, tag) in enumerate(timestamps):
        end = timestamps[i+1][0] if i+1 < len(timestamps) else duration
        tags.append({"start": start, "end": end, "tag": tag})

    return tags


def clean_filename(s):
    # Remove invalid characters and replace spaces with underscores
    return re.sub(r'[\\/*?:"<>|]', "", s).replace(" ", "_")

def split_audio(wav_path, tags, output_dir=r"D:/WSAI/data1/songs"):
    os.makedirs(output_dir, exist_ok=True)
    audio = AudioSegment.from_wav(wav_path)
    for tag in tags:
        start_ms = tag["start"] * 1000
        end_ms = tag["end"] * 1000
        seg = audio[start_ms:end_ms]
        fname = os.path.join(output_dir, f"{clean_filename(tag['tag'])}.wav")
        seg.export(fname, format="wav")
        print(f"Saved: {fname}")


# --- USAGE ---
youtube_url = "https://www.youtube.com/watch?v=U6cyXsQjzHY"
mp4_audio, duration, desc = download_audio(youtube_url)
convert_to_wav(mp4_audio, "audio.wav")
tag_data = extract_chapters(desc, duration)
split_audio("audio.wav", tag_data)
