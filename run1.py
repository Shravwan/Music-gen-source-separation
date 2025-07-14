import os
import subprocess
from pathlib import Path

COMMON_RAGAS = [
    "Kalyani", "Shankarabharanam", "Thodi", "Kharaharapriya", "Kamboji","Kambhoji", "Bhairavi", "Keeravani",
    "Hamsadhwani", "Mohanam", "Pantuvarali", "Bilahari", "Sankarabharanam", "Charukesi",
    "Mayamalavagowla", "Sahana", "Saveri", "Kaanada", "Anandabhairavi", "Hindolam", "Arabhi",
    "Harikambhoji", "Nattai", "Latangi", "Poorvikalyani", "Kapi", "Abhogi", "Begada", "Vasanta",
    "Ranjani", "Simhendramadhyamam", "Yadukula Kambhoji", "Varali", "Dhanyasi", "Manirangu",
    "Reetigowla", "Madhyamavathi", "Sri", "Sriranjani", "Jayamanohari", "Kedaragowla", "Suruti",
    "Kalyana Vasantham", "Shanmukhapriya", "Vagadheeswari", "Desh", "Kuntalavarali", "Durbar",
    "Devagandhari", "Gowlai", "Amritavarshini", "Suddha Dhanyasi", "Yamunakalyani",
    "Mukhari", "Mohana Kalyani", "Sarasangi", "Chakravakam", "Vachaspati", "Shuddha Saveri",
    "Shivaranjani", "Malavi", "Brindavani", "Suddha Seemanthini", "Natabhairavi", "Suddha Hindolam",
    "Udayaravichandrika", "Gambheera Nattai", "Gowrimanohari", "Narasimha Priya",
    "Kedaram", "Deshkar", "Neelambari", "Haripriya", "Khamas", "Kapinarayani", "Manjari",
    "Karnaranjani", "Kuranji", "Rasikapriya", "Rohini", "Madhuvanti", "Sindhubhairavi", "Behag",
    "Mishra Kapi", "Mandari", "Yamuna", "Revati", "Nayaki", "Chinthamani", "Rasali",
    "Chenchurutti", "Kalavati", "Janjuti", "Manolayam", "Sunadavinodini", "Kosalam",
    "Bhoopalam", "Revagupti", "Saraswathi", "Darbari Kanada", "Hameer Kalyani",
    "Saramati", "Asaveri", "Yogini", "Shivapriya", "Veenavadini", "Salaga Bhairavi",
    "Gowla", "Chalanata","Ragamalika","Kamalamanohari","Jonpuri","Bahudari"
]

def extract_raga(text):
    text = text.lower()
    for raga in COMMON_RAGAS:
        if raga.lower() in text:
            return raga
    return "Others"

module_dir = os.path.dirname(__file__)
source_sep_script = os.path.join(module_dir, "source_sep.py")
gender_split_script = os.path.join(module_dir, "gender_split.py")
extract_flute_script = os.path.join(module_dir, "extract_flute.py")
extract_violin_script = os.path.join(module_dir, "extract_violin.py")


input_folder = r"D:/WSAI/data1/songs"
output_base_dir = r"D:/WSAI/data1/split/by_raga"

for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".wav", ".mp3", ".flac")):
        continue

    input_path = os.path.join(input_folder, filename)
    song_name = Path(filename).stem
    raga = extract_raga(song_name)
    raga_dir = os.path.join(output_base_dir, raga)
    song_output_dir = os.path.join(raga_dir, song_name)

    os.makedirs(raga_dir, exist_ok=True)
    song_output_dir = os.path.join(raga_dir, song_name)
    vocals_raw = os.path.join(song_output_dir, "vocals.wav")
    others_raw = os.path.join(song_output_dir, "others.wav")
    expected_vocals = os.path.join(song_output_dir, f"{song_name}_vocals.wav")
    expected_others = os.path.join(song_output_dir, f"{song_name}_other.wav")
    
    if os.path.exists(expected_vocals) and os.path.exists(expected_others):
        print(f"Skipping {song_name} (already fully processed)")
        continue

    
    if not (os.path.exists(vocals_raw) and os.path.exists(others_raw)):
        print(f"\n==== Processing: {song_name} ====")
        print("Running source separation...")
        result = subprocess.run(["python", source_sep_script, input_path, song_output_dir], capture_output=True, text=True)
        print(result.stdout)

        
        vocal_path = None
        others_path = None
        for line in result.stdout.splitlines():
            if line.startswith("VOCALS_FILE="):
                vocal_path = line.split("=", 1)[1].strip()
            elif line.startswith("OTHERS_FILE="):
                others_path = line.split("=", 1)[1].strip()

        if not vocal_path or not os.path.exists(vocal_path):
            print("Vocals file not found, skipping.")
            continue
        if not others_path or not os.path.exists(others_path):
            print("Others file not found, skipping.")
            continue
    else:
        print(f"Stems already found for {song_name}, skipping separation.")
        vocal_path = vocals_raw
        others_path = others_raw


    
    print("Running gender vocal split...")
    env = os.environ.copy()
    env["SPEECHBRAIN_LOCAL_DOWNLOAD_STRATEGY"] = "copy"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    subprocess.run(["python", gender_split_script, vocal_path, song_output_dir], check=True, env=env)

    
    song_name = Path(filename).stem

    flute_path = os.path.join(song_output_dir, f"{song_name}_flute.wav")
    violin_path = os.path.join(song_output_dir, f"{song_name}_violin.wav")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(module_dir, "..")) 
     # add D:\WSAI
    print("Extracting flute...")
    subprocess.run(["python", extract_flute_script], check=True, env=env)


    print("Extracting violin...")
    subprocess.run(["python", extract_violin_script], check=True,env=env)


    print(f"Finished processing {song_name}. Output at: {song_output_dir}")
