if __name__ == "__main__":
    # Get the path of the most recently modified mp3 file in the data directory
    import glob
    import os
    from pathlib import Path

    from rich import print

    from Whisper.formatter import llm_format
    from Whisper.utils import result_to_txt, s2hk
    from Whisper.whisper_hf import whisper_hf

    DATA_DIR = Path("data")
    mp3_files = glob.glob(str(DATA_DIR / "*.mp3"))

    # Sort files by modification time (most recent first)
    latest_audio = max(mp3_files, key=os.path.getmtime)
    print(f"Using most recent audio file: {latest_audio}")
    result = whisper_hf(latest_audio)

    subtitle = result_to_txt(result)
    subtitle = s2hk(subtitle)
    subtitle = llm_format(subtitle)
    print(subtitle[:1000])
