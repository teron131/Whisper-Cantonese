"""
Somehow always some videos are skipped. Likely due to the web calls and rate limits.
This script will keep running until no more new videos are found.
Make sure to run without interrupting the script so that the csv are correctly updated.
If it is detected as bot, retry with different network (e.g. reconnect with VPN).
"""

import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

from pytubefix import Caption, Channel, YouTube
from pytubefix.exceptions import VideoUnavailable
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

DATA_DIR = Path("data_en")
DATA_DIR.mkdir(exist_ok=True)
CAPTION_LANGUAGES = ["en", "en.j3PyPqV-e1s"]


@dataclass
class ChannelInfo:
    """Class for keeping track of YouTube channel info."""

    channel_id: str
    channel_dir: Path
    channel_name: str
    video_urls: List[str]
    increase_threshold: int


def atomic_download(
    download_func: Callable[[Path], None],
    output_dir: Path,
    filename: str,
) -> bool:
    """Download a file atomically.

    Using a temporary file to avoid race conditions.

    If the download fails, the temporary file will be cleaned up.

    If the download succeeds, the temporary file will be moved to the final path.

    Args:
        download_func: A function that takes a Path object and downloads the file.
        output_dir: The directory to save the file to.
        filename: The name of the file to download.

    Returns:
        bool: True if the download was successful, False otherwise.
    """
    temp_filename = filename + ".tmp"
    final_path = output_dir / filename
    temp_path = output_dir / temp_filename
    if final_path.exists():
        tqdm.write(f"Already exists: {final_path}")
        return True

    success = False
    try:
        download_func(temp_path)
        if not temp_path.exists():
            tqdm.write(f"Failed to download {filename}")
        else:
            os.replace(temp_path, final_path)
            tqdm.write(f"Downloaded: {final_path}")
            success = True
    except Exception as e:
        tqdm.write(f"Error downloading {filename}: {e}")
    finally:
        # Clean up temporary file if download was not successful
        if not success and temp_path.exists():
            try:
                temp_path.unlink()
                tqdm.write(f"Cleaned up temporary file: {temp_path}")
            except Exception as cleanup_error:
                tqdm.write(f"Error cleaning up temporary file {temp_path}: {cleanup_error}")
    return success


def download_audio(
    video: YouTube,
    output_dir: Path = DATA_DIR,
    title_filename: bool = False,
) -> bool:
    """Download the audio from a YouTube video.

    Args:
        video: The YouTube video object
        output_dir: The directory to save the file to
        title_filename: Whether to use the title as the filename

    Returns:
        bool: True if the download was successful, False otherwise.
    """
    filename = f"{video.video_id}.m4a" if not title_filename else f"{video.title}.m4a"

    def download_func(temp_path: Path):
        # Use the temporary file's name (which includes the .tmp suffix) for downloading
        video.streams.get_audio_only().download(output_path=str(output_dir), filename=temp_path.name)

    return atomic_download(download_func, output_dir, filename)


def download_caption(
    video: YouTube,
    output_dir: Path = DATA_DIR,
    title_filename: bool = False,
    target_caption: Caption = None,
) -> bool:
    """Download the caption from a YouTube video.

    Args:
        video: The YouTube video object
        output_dir: The directory to save the file to
        title_filename: Whether to use the title as the filename
        target_caption: The caption to download

    Returns:
        bool: True if the download was successful, False otherwise.
    """
    filename = f"{video.video_id}.srt" if not title_filename else f"{video.title}.srt"
    if target_caption is None:
        captions = video.captions
        target_caption = next((captions.get(language) for language in CAPTION_LANGUAGES if language in captions), None)

    def download_func(temp_path: Path):
        target_caption.save_captions(filename=str(temp_path))

    return atomic_download(download_func, output_dir, filename)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
def download_audio_caption(
    url: str,
    channel_info: ChannelInfo,
) -> dict | None:
    """Download both the audio and caption if any of the preferred languages are available. Return the video metadata if successful.

    Both audio and caption are either downloaded or skipped together.

    Args:
        url (str): The URL of the video
        channel_info (ChannelInfo): Information about the channel

    Returns:
        dict | None: Video metadata if successful, None otherwise
    """
    video = YouTube(url)
    try:
        captions = video.captions
        caption: Caption = next((captions.get(language) for language in CAPTION_LANGUAGES if language in captions), None)

        # Skip videos without captions in preferred languages
        if caption is None:
            return None

        output_dir = channel_info.channel_dir

        if not download_audio(video=video, output_dir=output_dir, title_filename=False):
            # Skip the video if audio is not available
            return None

        if not download_caption(video=video, output_dir=output_dir, title_filename=False, target_caption=caption):
            # Skip the video if caption is not available
            audio_file = output_dir / f"{video.video_id}.m4a"
            if audio_file.exists():
                audio_file.unlink()
            return None

        # Return video metadata
        return {
            "video_id": video.video_id,
            "title": video.title,
            "watch_url": video.watch_url,
            "length": video.length,
            "caption_code": caption.code,
        }

    except VideoUnavailable:
        tqdm.write(f"Video {video.watch_url} is unavailable, skipping.")
        return None


# -------------------------------------------------------------------------------------------------


def process_channel(channel_url: str) -> ChannelInfo:
    """Get all video URLs from a YouTube channel.

    Args:
        channel_url: The URL of the YouTube channel

    Returns:
        ChannelInfo: Object containing channel information and video URLs
    """
    channel = Channel(channel_url)
    channel_id = channel.channel_uri.replace("/@", "").lower()  # Must be lower case
    channel_dir = DATA_DIR / channel_id
    channel_dir.mkdir(exist_ok=True)

    # Get all video URLs first
    videos = list(channel.videos)
    video_urls = [video.watch_url for video in videos]
    tqdm.write(f"Collected {len(video_urls)} videos from channel {channel.channel_name}")

    # Filter out the videos already downloaded
    existing_videos = set()
    if channel_dir.exists():
        existing_videos = {file.stem for file in channel_dir.glob("*.m4a")}

    # Filter out videos that are already processed
    filtered_video_urls = [url for url in video_urls if url.split("v=")[1] not in existing_videos]

    tqdm.write(f"Filtered out {len(existing_videos)} videos that already exist in {channel_dir}")
    tqdm.write(f"Processing {len(filtered_video_urls)} videos from {channel.channel_name}")

    return ChannelInfo(
        channel_id=channel_id,
        channel_dir=channel_dir,
        channel_name=channel.channel_name,
        video_urls=filtered_video_urls,
        increase_threshold=int(len(filtered_video_urls) * 0.01),
    )


def download_videos(channel_info: ChannelInfo) -> List[Dict[str, Any]]:
    """Process videos in parallel using ThreadPoolExecutor.

    Args:
        channel_info: Information about the channel

    Returns:
        list: List of dictionaries containing video metadata
    """
    if len(channel_info.video_urls) == 0:
        tqdm.write("No new videos to download.")
        return []

    results = []
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 3) as executor:
        # Submit URL processing tasks
        futures = [executor.submit(download_audio_caption, url, channel_info) for url in channel_info.video_urls]

        # Process results as they complete
        with tqdm(total=len(channel_info.video_urls), desc="Processing videos", unit="video") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
                pbar.update(1)

    return results


def save_results(results: List[Dict[str, Any]], channel_info: ChannelInfo) -> int:
    """Save results to a CSV file.

    Args:
        results: List of dictionaries containing video metadata
        channel_info: Information about the channel

    Returns:
        int: Number of new videos saved
    """
    if not results:
        tqdm.write("No new videos to download. Stopping.")
        return 0

    csv_path = channel_info.channel_dir / "metadata.csv"

    # Read existing entries to avoid duplicates
    existing_entries = set()
    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                existing_entries.add(row["watch_url"])

    # Filter out results that already exist in the CSV
    new_results = [result for result in results if result["watch_url"] not in existing_entries]

    # Check if we have any results before trying to write to CSV
    if new_results:
        with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Only write headers if the file is new (empty)
            if csvfile.tell() == 0:
                writer.writeheader()

            for result in new_results:
                writer.writerow(result)

        tqdm.write(f"Saved {len(new_results)} new videos with captions to {csv_path}")
        tqdm.write(f"Total videos: {len(existing_entries) + len(new_results)}")

    return len(new_results)


def download_channel(channel_url: str) -> tuple[int, ChannelInfo]:
    """Download all videos from a YouTube channel and save the results to a CSV file.

    Args:
        channel_url (str): The URL of the YouTube channel

    Returns:
        int: Number of new videos downloaded
    """
    # Get channel info
    channel_info = process_channel(channel_url)

    # Download videos in parallel
    results = download_videos(channel_info)

    # Save results to CSV
    new_videos = save_results(results, channel_info)

    return new_videos, channel_info


if __name__ == "__main__":
    channel_urls = [
        ## "https://www.youtube.com/@3blue1brown",
        ## "https://www.youtube.com/@BoxofficeMoviesScenes",
        "https://www.youtube.com/@business",
        "https://www.youtube.com/@TheInfographicsShow",
        "https://www.youtube.com/@marvel",
        ## "https://www.youtube.com/@MarkRober",
        "https://www.youtube.com/@mitocw",
        ## "https://www.youtube.com/@mkbhd",
        ## "https://www.youtube.com/@MSFTMechanics",
        ## "https://www.youtube.com/@neoexplains",
        ## "https://www.youtube.com/@NVIDIA",
        ## "https://www.youtube.com/@QuantaScienceChannel",
        ## "https://www.youtube.com/@TEDEd/",
        ## "https://www.youtube.com/@TwoMinutePapers",
        ## "https://www.youtube.com/@veritasium",
    ]

    if channel_urls == []:
        channel_url = input("Please enter the YouTube channel URL: ")
        channel_urls.append(channel_url)

    for channel_url in channel_urls:
        MAX_RETRIES = 5
        retries = 0
        new_videos = -1
        while new_videos != 0 and retries < MAX_RETRIES:
            new_videos, channel_info = download_channel(channel_url)
            # Stop if new videos do not exceed the threshold
            if new_videos <= channel_info.increase_threshold:
                tqdm.write(f"Downloaded only {new_videos} new videos, below threshold {channel_info.increase_threshold}, stopping for channel {channel_url}")
                break
            retries += 1
