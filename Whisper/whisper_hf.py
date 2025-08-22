import timeit
import warnings
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

warnings.filterwarnings("ignore")

load_dotenv()


def whisper_hf(
    audio: str | bytes,
    model_id: str = "openai/whisper-large-v3-turbo",
) -> dict:
    """Transcribe audio file using whisper-large-v3-turbo model with Hugging Face optimization.

    Args:
        audio (str | bytes): The audio file path string or bytes data to be transcribed.

    Returns:
        dict: A dictionary containing the transcription result with the following structure:
            {
                "text": str,    # Full transcribed text
                "chunks": [     # List of transcription chunks
                    # Each chunk is a dictionary with:
                    {
                        "timestamp": tuple[float],  # Start and end time of the chunk
                        "text": str,               # Transcribed text for this chunk
                    },
                ]
            }
    """
    start_time = timeit.default_timer()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device: {device} ({torch_dtype})")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        use_safetensors=True,
    ).to(device)
    # model.config.attn_implementation = "sdpa"  # It only works this way
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    result = pipe(audio, generate_kwargs={"language": "yue"})
    end_time = timeit.default_timer()
    print(f"Time taken: {end_time - start_time} seconds")
    return result


if __name__ == "__main__":
    # Get the path of the most recently modified mp3 file in the data directory
    import glob
    import os
    from formatter import llm_format
    from pathlib import Path
    from typing import cast

    from rich import print
    from utils import result_to_txt, s2hk
    from whisper_hf import whisper_hf

    data_dir = Path("dataset")

    files = []
    for dirpath, dirs, filenames in os.walk(data_dir):
        if not dirs:  # Only process directories with no subdirectories (deepest tier)
            subdir = Path(dirpath)
            for file in filenames:
                files.append(subdir / file)

    audio_files = [file for file in files if cast(Path, file).suffix == ".mp3"]

    # Sort files by modification time (most recent first)
    # latest_audio = str(max(audio_files, key=os.path.getmtime))
    latest_audio = audio_files[0]
    print(f"Using most recent audio file: {latest_audio}")
    result = whisper_hf(str(latest_audio))

    subtitle = result_to_txt(result)
    subtitle = s2hk(subtitle)
    # subtitle = llm_format(subtitle)
    print(subtitle[:1000])
