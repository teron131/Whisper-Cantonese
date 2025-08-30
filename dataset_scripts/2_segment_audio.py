import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import cast

import pysrt
from pysrt import SubRipItem, SubRipTime
from tqdm import tqdm

DATA_DIR = Path("data_en")
OUTPUT_DATA_DIR = Path("dataset_en")

THRESHOLD: float = 30.0
BUFFER: float = 0.5

files = []
for dirpath, dirs, filenames in os.walk(DATA_DIR):
    if not dirs:  # Only process directories with no subdirectories (deepest tier)
        subdir = Path(dirpath)
        for file in filenames:
            files.append(subdir / file)

audio_files = [file for file in files if cast(Path, file).suffix == ".m4a"]


def get_duration(audio_file: Path) -> float:
    """Get the accurate audio file duration in float seconds, or None if ffprobe fails.

    Args:
        audio_file: The audio file to get the duration of

    Returns:
        The duration of the audio file in float seconds
    """
    try:
        command = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(audio_file)]
        proc = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(proc.stdout.strip())
    except Exception:
        raise ValueError(f"Failed to get audio duration for {audio_file}")


@dataclass
class FileInfo:
    """Information of an audio file to be processed."""

    audio_file: Path
    caption_file: Path
    channel_dir: str
    audio_filename: str
    channel_metadata_file: Path
    output_dir: Path
    full_duration: float

    def __init__(self, audio_file: Path):
        self.audio_file = audio_file
        self.caption_file = audio_file.with_suffix(".srt")
        self.channel_dir = audio_file.parts[-2]  # Channel ID could contain '.'
        self.audio_filename = audio_file.stem
        self.channel_metadata_file = DATA_DIR / self.channel_dir / "metadata.csv"
        self.output_dir = OUTPUT_DATA_DIR / self.channel_dir / self.audio_filename
        self.full_duration = get_duration(self.audio_file)


@dataclass
class GroupInfo:
    """Information of a group of subtitles to be processed together."""

    file_info: FileInfo
    subs: list[SubRipItem]
    channel_metadata: dict
    first_idx: int
    last_idx: int
    group_name: str
    output_audio_path: Path
    output_metadata_path: Path
    lock_file: Path

    def __init__(self, file_info: FileInfo, subs: list[SubRipItem], channel_metadata: dict):
        self.file_info = file_info
        self.subs = subs
        self.channel_metadata = channel_metadata
        self.first_idx = subs[0].index
        self.last_idx = subs[-1].index
        self.group_name = f"{self.first_idx}" if self.first_idx == self.last_idx else f"{self.first_idx}-{self.last_idx}"
        base = file_info.output_dir / f"{file_info.audio_filename}_{self.group_name}"
        self.output_audio_path = base.with_suffix(".mp3")
        self.output_metadata_path = base.with_suffix(".json")
        self.lock_file = base.with_suffix(".lock")

    @property
    def exists(self) -> bool:
        """Check if both the output audio and valid metadata JSON exist."""
        return self.output_audio_path.exists() and _is_valid_json(self.output_metadata_path)

    @property
    def group_timestamps(self) -> tuple[float, float, float]:
        """Get padded start time, padded end time, and duration of the entire group in seconds."""
        raw_start = _subtime_to_seconds(self.subs[0].start)
        raw_end = _subtime_to_seconds(self.subs[-1].end)
        # Apply buffer
        start = max(0.0, raw_start - BUFFER)
        end = min(raw_end + BUFFER, self.file_info.full_duration)
        return start, end, end - start

    @property
    def metadata_dict(self) -> dict:
        """Convert the group to a metadata dictionary."""
        start, end, clip_duration = self.group_timestamps
        return {
            "caption": " ".join(sub.text for sub in self.subs),
            "channel_id": self.file_info.channel_dir,
            "video_id": self.file_info.audio_filename,
            "title": self.channel_metadata.get("title", ""),
            "url": self.channel_metadata.get("watch_url", ""),
            "caption_code": self.channel_metadata.get("caption", ""),
            "full_duration": self.file_info.full_duration,
            "clip_start": start,
            "clip_end": end,
            "clip_duration": clip_duration,
            "segments": [
                {
                    "segment_index": sub.index,
                    "start_time": str(sub.start.to_time()),
                    "end_time": str(sub.end.to_time()),
                    "text": sub.text,
                }
                for sub in self.subs
            ],
        }


def _subtime_to_seconds(sub: SubRipTime) -> float:
    """Convert a datetime.time object to seconds."""
    t: time = sub.to_time()
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def _is_valid_json(path: Path) -> bool:
    """Return True if `path` is a non-empty file containing valid JSON."""
    try:
        if path.stat().st_size == 0:
            return False
        with open(path, "r", encoding="utf-8") as fp:
            json.load(fp)
        return True
    except Exception:
        return False


def group_subtitles(subs: list[SubRipItem]) -> list[list[SubRipItem]]:
    """Group subtitles into buckets with time-span up to threshold seconds after padding each subtitle by buffer.

    Args:
        subs: List of SubRipItem objects

    Returns:
        List of lists of SubRipItem objects
    """
    # Filter out subtitles that are too long (abnormal)
    subs = [sub for sub in subs if _subtime_to_seconds(sub.duration) + 2 * BUFFER <= THRESHOLD]
    if not subs:
        return []

    groups: list[list[SubRipItem]] = []
    group: list[SubRipItem] = [subs[0]]
    start = _subtime_to_seconds(subs[0].start)
    for sub in subs[1:]:
        end = _subtime_to_seconds(sub.end)
        # Check padded duration of the group
        if end - start + 2 * BUFFER <= THRESHOLD:
            group.append(sub)
        else:
            groups.append(group)
            group = [sub]
            start = _subtime_to_seconds(sub.start)
    groups.append(group)
    return groups


def read_channel_metadata(file_info: FileInfo) -> dict:
    """Read the channel metadata for the given audio file from the raw dataset.

    Args:
        file_info: FileInfo object containing the audio file and metadata

    Returns:
        dict: The channel metadata
    """
    with open(file_info.channel_metadata_file, encoding="utf-8") as fp:
        return next((row for row in csv.DictReader(fp) if row.get("video_id") == file_info.audio_filename), {})


def _try_acquire_lock(lock_file: Path) -> bool:
    """Attempt to create a lockfile; return True if successful."""
    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def _release_lock(lock_file: Path) -> None:
    """Remove the lockfile if it exists."""
    try:
        lock_file.unlink()
    except OSError:
        pass


def process_group(audio_file: Path, group_info: GroupInfo) -> None:
    """Process a group of subtitles and audio file.

    Args:
        audio_file: The audio file to process
        group_info: GroupInfo object containing the group of subtitles
    """
    if group_info.exists or not _try_acquire_lock(group_info.lock_file):
        return
    try:
        # 1) Pre-seek into the file and only decode the group's duration
        group_start, _, group_duration = group_info.group_timestamps

        # Build padded intervals for each subtitle and merge overlapping intervals
        intervals: list[tuple[float, float]] = []
        for sub in group_info.subs:
            sub_start = _subtime_to_seconds(sub.start)
            sub_end = _subtime_to_seconds(sub.end)
            padded_start = max(0.0, sub_start - BUFFER)
            padded_end = sub_end + BUFFER
            intervals.append((padded_start, padded_end))
        intervals.sort(key=lambda x: x[0])
        merged_intervals: list[tuple[float, float]] = []
        curr_start, curr_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= curr_end:
                curr_end = max(curr_end, end)
            else:
                merged_intervals.append((curr_start, curr_end))
                curr_start, curr_end = start, end
        merged_intervals.append((curr_start, curr_end))

        # Build ffmpeg filter_complex with merged intervals; clamp times and use duration to avoid negative durations
        trims: list[str] = []
        valid_count = 0
        for start, end in merged_intervals:
            # Clamp relative start and end to input segment
            start_rel = max(0.0, start - group_start)
            end_rel = min(end - group_start, group_duration)
            duration_rel = end_rel - start_rel
            if duration_rel <= 0:
                tqdm.write(f"[WARN] Skipping empty or invalid interval for group {group_info.group_name}: start={start_rel}, duration={duration_rel}")
                continue
            trims.append(f"[0:a]atrim=start={start_rel}:duration={duration_rel},asetpts=PTS-STARTPTS[a{valid_count}]")
            valid_count += 1
        if valid_count == 0:
            tqdm.write(f"[WARN] No valid audio intervals for group {group_info.group_name}, skipping")
            return
        inputs = "".join(f"[a{i}]" for i in range(valid_count))
        filter_complex = ";".join(trims + [f"{inputs}concat=n={valid_count}:v=0:a=1[out]"])

        # SPEED UP FFmpeg: fast‐seek into group_start, only decode group_duration
        segment_command = [
            "ffmpeg",
            "-y",
            "-ss",
            str(group_start),
            "-i",
            str(audio_file),
            "-t",
            str(group_duration),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-c:a",
            "libmp3lame",
            "-q:a",
            "5",
            "-ar",
            "16000",
            str(group_info.output_audio_path),
        ]
        subprocess.run(segment_command, check=True, capture_output=True, text=True)

        # Write metadata atomically
        tmp_path = group_info.output_metadata_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as fp:
            json.dump(group_info.metadata_dict, fp, ensure_ascii=False, indent=2)
            os.fsync(fp.fileno())
        os.replace(tmp_path, group_info.output_metadata_path)

    except subprocess.CalledProcessError as e:
        tqdm.write(f"[ERROR] FFmpeg failed for {group_info.output_audio_path}: {e.stderr}")
    except Exception as e:
        tqdm.write(f"[ERROR] Unexpected error for group {group_info.group_name}: {e}")
    finally:
        _release_lock(group_info.lock_file)


def segment_audio(audio_file: Path) -> None:
    """Split an audio file into ~30s buckets of adjacent subtitles.

    Args:
        audio_file: The audio file to process
    """
    file_info = FileInfo(audio_file)
    file_info.output_dir.mkdir(parents=True, exist_ok=True)

    subs = cast(list[SubRipItem], pysrt.open(file_info.caption_file))
    groups = group_subtitles(subs)
    channel_metadata = read_channel_metadata(file_info)

    group_infos = [gi for gi in (GroupInfo(file_info, g, channel_metadata) for g in groups) if not gi.exists]

    if not group_infos:
        tqdm.write(f"[SKIP] All segments done for {file_info.audio_filename}")
        return

    # Dispatch only pending jobs
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_group, audio_file, gi) for gi in group_infos]
        list(tqdm(as_completed(futures), total=len(futures), desc=f"Processing {file_info.audio_filename} segments"))


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        futures = [executor.submit(segment_audio, af) for af in audio_files]
        list(tqdm(as_completed(futures), total=len(futures), desc="Processing audio files", miniters=1))
