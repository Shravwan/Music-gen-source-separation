import os
import sys
import torch
import torchaudio
import librosa
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

class CarnaticSourceSeparator:
    """
    A comprehensive source separation system for Carnatic music 
    supporting both Demucs and Spleeter models.
    """

    def __init__(self, model_type='demucs', model_name='htdemucs'):
        """
        Initialize the source separator.

        Args:
            model_type (str): 'demucs' or 'spleeter'
            model_name (str): specific model to use (e.g., 'htdemucs', 'hdemucs_mmi')
        """
        self.model_type = model_type
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        if model_type == 'demucs':
            self._setup_demucs()

    def _setup_demucs(self):
        """Setup Demucs model for source separation."""
        try:
            from demucs.pretrained import get_model
            from demucs.apply import apply_model

            print(f"Loading Demucs model: {self.model_name}")
            self.model = get_model(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.apply_model = apply_model
            print("Demucs model loaded successfully!")

        except Exception as e:
            print(f"Error loading Demucs: {e}")
            print("Falling back to simpler approach...")
            self._setup_demucs_alternative()

    def _setup_demucs_alternative(self):
        """Alternative Demucs setup using subprocess for command line interface."""
        import subprocess
        try:
            # Test if demucs command is available
            result = subprocess.run(['demucs', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.use_subprocess = True
                print("Using Demucs command line interface")
            else:
                raise Exception("Demucs command not found")
        except Exception as e:
            print(f"Demucs setup failed: {e}")
            raise


    def load_audio(self, file_path, target_sr=44100):
        """
        Load audio file and ensure proper format.

        Args:
            file_path (str): Path to audio file
            target_sr (int): Target sample rate

        Returns:
            tuple: (audio_tensor, sample_rate)
        """
        print(f"Loading audio: {file_path}")

        # Load with librosa for better format support
        audio, sr = librosa.load(file_path, sr=target_sr, mono=False)

        # Ensure stereo format
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        elif audio.shape[0] > 2:
            audio = audio[:2]  # Take first two channels

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()

        print(f"Audio loaded: shape={audio_tensor.shape}, sr={sr}")
        return audio_tensor, sr

    def separate_with_demucs(self, audio_tensor, sr):
        """
        Perform source separation using Demucs.

        Args:
            audio_tensor: Input audio tensor
            sr: Sample rate

        Returns:
            dict: Dictionary with separated sources
        """
        print("Separating sources with Demucs...")

        # Ensure audio is on the correct device
        audio_tensor = audio_tensor.to(self.device)

        with torch.no_grad():
            # Apply the model
            sources = self.apply_model(self.model, audio_tensor[None], device=self.device)[0]

        # Get source names from the model
        source_names = self.model.sources

        # Convert to dictionary
        separated = {}
        mapping = {
        "drums": "mridangam",
        "vocals": "vocals",
        "other": "veena_violin"
        }

        for i, name in enumerate(source_names):
            if name == "bass":
                continue  # Skip drums
            if name in mapping:
                new_name = mapping[name]
                separated[name] = sources[i].cpu().numpy()

        print(f"Separated sources: {list(separated.keys())}")
        return separated

    def save_separated_sources(self, separated_sources, output_dir, original_filename, sr=44100):
        """
        Save separated audio sources as individual WAV files.

        Args:
            separated_sources (dict): Dictionary of separated sources
            output_dir (str): Output directory
            original_filename (str): Original filename for naming
            sr (int): Sample rate
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get base filename without extension
        base_name = Path(original_filename).stem

        saved_files = []

        for source_name, audio_data in separated_sources.items():
            # Construct output filename
            output_filename = f"{base_name}_{source_name}.wav"
            output_file_path = output_path / output_filename

            # Ensure audio data is in correct format
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.numpy()

            # Handle different audio shapes
            if audio_data.ndim == 3:  # (batch, channels, time)
                audio_data = audio_data[0]

            if audio_data.ndim == 2 and audio_data.shape[0] == 2:  # (channels, time)
                # Transpose to (time, channels) for librosa
                audio_data = audio_data.T
            elif audio_data.ndim == 1:  # mono
                pass  # Keep as is

            # Save using librosa
            try:
                if audio_data.ndim == 2:
                    # Stereo
                    librosa.output.write_wav(str(output_file_path), audio_data.T, sr)
                else:
                    # Mono
                    librosa.output.write_wav(str(output_file_path), audio_data, sr)

                print(f"Saved: {output_file_path}")
                saved_files.append(str(output_file_path))

            except Exception as e:
                print(f"Error saving {source_name}: {e}")
                # Try alternative saving method
                try:
                    import soundfile as sf
                    sf.write(str(output_file_path), audio_data, sr)
                    print(f"Saved with soundfile: {output_file_path}")
                    saved_files.append(str(output_file_path))
                except:
                    print(f"Failed to save {source_name} with any method")

        return saved_files

    def process_carnatic_song(self, input_file, output_dir):
        """
        Complete pipeline to separate a Carnatic song into instruments.

        Args:
            input_file (str): Path to input audio file
            output_dir (str): Directory to save separated files

        Returns:
            list: List of saved file paths
        """
        print(f"\n{'='*50}")
        print(f"Processing Carnatic Song: {input_file}")
        print(f"{'='*50}")

        try:
                # Load audio
            audio_tensor, sr = self.load_audio(input_file)

                # Separate sources
            separated_sources = self.separate_with_demucs(audio_tensor, sr)

            # Save separated sources
            base_name = Path(input_file).stem
            song_folder = Path(output_dir) / base_name
            saved_files = self.save_separated_sources(
                separated_sources, song_folder, input_file, sr
            )

            print(f"\nSuccessfully separated {len(separated_sources)} sources:")
            for i, (source, _) in enumerate(separated_sources.items(), 1):
                print(f"  {i}. {source}")

            print(f"\nFiles saved to: {output_dir}")

            return saved_files

        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            return []


# Example usage function
def separate_carnatic_music(input_file, output_dir="separated_audio"):
    """
    High-level function to separate Carnatic music.

    Args:
        input_file (str): Path to the Carnatic song file
        output_dir (str): Directory to save separated instruments
        model (str): Model to use - "demucs" or "spleeter"

    Returns:
        list: Paths to the separated audio files
    """

    # Available Demucs models:
    # - 'htdemucs': Latest hybrid transformer model (recommended)
    # - 'hdemucs_mmi': Hybrid Demucs v3
    # - 'mdx': Trained on MusDB HQ
    # - 'mdx_extra': Trained with extra data

    separator = CarnaticSourceSeparator(model_type='demucs', model_name='htdemucs')

    return separator.process_carnatic_song(input_file, output_dir)


# Additional utility functions for Carnatic music specific processing
def enhance_for_carnatic_music(audio_file, output_file):
    """
    Apply Carnatic music specific preprocessing.
    This can help improve separation quality for Indian classical music.
    """
    print("Applying Carnatic music enhancements...")

    # Load audio
    y, sr = librosa.load(audio_file, sr=44100)

    # Apply spectral gating to reduce noise (helps with tanpura drone)
    import noisereduce as nr
    y_reduced = nr.reduce_noise(y=y, sr=sr)

    # Enhance harmonics (important for Carnatic music)
    y_harmonic, y_percussive = librosa.effects.hpss(y_reduced, margin=3.0)

    # Recombine with emphasis on harmonics
    y_enhanced = 0.7 * y_harmonic + 0.3 * y_percussive + 0.1 * y_reduced

    # Save enhanced audio
    librosa.output.write_wav(output_file, y_enhanced, sr)
    print(f"Enhanced audio saved: {output_file}")

    return output_file


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python source_sep1.py <input_file> <output_dir>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_base_dir = sys.argv[2]

    separator = CarnaticSourceSeparator(model_type='demucs', model_name='htdemucs')
    separated_files = separator.process_carnatic_song(input_path, output_base_dir)

    vocals_file = None
    others_file = None

    for file in separated_files:
        name = Path(file).stem.lower()
        if "vocals" in name:
            vocals_file = file
        elif "veena_violin" in name or "other" in name:
            others_file = file

    if vocals_file:
        print(f"VOCALS_FILE={vocals_file}")
    if others_file:
        print(f"OTHERS_FILE={others_file}")



