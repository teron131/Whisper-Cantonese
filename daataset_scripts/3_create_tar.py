import json
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

# 1 GiB
THRESHOLD_BYTES = 1 * 1024**3
INPUT_DIR = Path("dataset_en")
OUTPUT_DIR = Path("dataset_hf_en")


def make_shard(
    shard_index: int,
    json_paths: list[Path],
    audio_paths: list[Path],
    shard_out: Path,
) -> None:
    """
    Create a shard of the dataset.

    Args:
        shard_index: The index of the shard to create.
        json_paths: A list of paths to the JSON files in the shard.
        audio_paths: A list of paths to the audio files in the shard.
        shard_out: The path to the output shard file.
    """
    with tarfile.open(shard_out, "w") as tar:
        for json_path, audio_path in zip(json_paths, audio_paths):
            # 1) Verify that JSON is valid & non‑empty
            try:
                with open(json_path, "r", encoding="utf‑8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[warning] skipping corrupt metadata {json_path}: {e}")
                continue

            # 2) Add the valid files
            tar.add(audio_path, arcname=audio_path.name)
            tar.add(json_path, arcname=json_path.name)


def shard_channel(
    channel_dir: Path,
    out_channel_dir: Path,
    threshold: int,
) -> None:
    """
    Gather all .mp3/.json pairs under channel_dir (recursively),
    then split them into tar shards of roughly 'threshold' bytes.

    Args:
        channel_dir: The directory containing the audio and JSON files.
        out_channel_dir: The directory to write the shards to.
        threshold: The maximum size of a shard in bytes.
    """
    # 1) Collect all valid pairs
    pairs = []
    for mp3_file in sorted(channel_dir.rglob("*.mp3")):
        json_file = mp3_file.with_suffix(".json")
        if json_file.exists():
            pairs.append((mp3_file, json_file))
    if not pairs:
        return

    # 2) Bucket into shards
    shards = []
    current = []
    current_size = 0
    for mp3_file, json_file in pairs:
        pair_size = mp3_file.stat().st_size + json_file.stat().st_size
        # if adding this pair would overflow and we already have some in 'current', flush
        if current and current_size + pair_size > threshold:
            shards.append(current)
            current = []
            current_size = 0
        current.append((mp3_file, json_file))
        current_size += pair_size
    if current:
        shards.append(current)

    # 3) Write out each shard
    channel_id = channel_dir.name
    out_channel_dir.mkdir(parents=True, exist_ok=True)
    for idx, shard in enumerate(shards):
        tar_path = out_channel_dir / f"{channel_id}_{idx:03d}.tar"
        if tar_path.exists():
            continue  # Skip already-built shard

        json_paths = [json_file for _, json_file in shard]
        audio_paths = [mp3_file for mp3_file, _ in shard]
        make_shard(idx, json_paths, audio_paths, tar_path)


def process_channel(args: tuple[Path, Path, int]) -> None:
    """
    Process a channel directory and create shards of the dataset.

    Args:
        args: A tuple containing the channel directory, output directory, and threshold.
    """
    channel_dir, output_dir, threshold = args
    out_channel = output_dir / channel_dir.name
    shard_channel(channel_dir, out_channel, threshold)


def main() -> None:
    if not INPUT_DIR.exists():
        print(f"⚠️  Input directory {INPUT_DIR!r} not found, nothing to do.")
        return

    # Process all channel directories directly under the input directory
    channel_dirs = sorted(d for d in INPUT_DIR.iterdir() if d.is_dir())
    if not channel_dirs:
        print(f"[SKIP] No channels under '{INPUT_DIR}/', skipping.")
        return

    desc = "Sharding channels"
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        args = [(channel_dir, OUTPUT_DIR, THRESHOLD_BYTES) for channel_dir in channel_dirs]
        list(tqdm(executor.map(process_channel, args), total=len(args), desc=desc))


if __name__ == "__main__":
    main()
