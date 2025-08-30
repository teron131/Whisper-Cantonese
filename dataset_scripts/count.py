import os
from pathlib import Path

import pandas as pd


def read_channel_csv_files(data_dir: str = "data") -> dict:
    """Read all CSV files in channel subdirectories and return them as a dictionary.

    Structure:
        {channel_name: {csv_file_stem: DataFrame, ...}, ...}
    """
    data_dir = Path(data_dir)
    results = {}
    if not data_dir.exists():
        return results

    for channel_dir in data_dir.iterdir():
        if channel_dir.is_dir():
            channel_csvs = {}
            for csv_file in channel_dir.glob("*.csv"):
                try:
                    channel_csvs[csv_file.stem] = pd.read_csv(csv_file)
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
            if channel_csvs:
                results[channel_dir.name] = channel_csvs

    return results


def calculate_channel_statistics(channel_data: dict) -> None:
    """Calculate video statistics for each channel and print a formatted summary."""
    total_videos = 0
    total_seconds = 0
    channel_stats = {}

    # Compute stats per channel
    for channel_name, files in channel_data.items():
        videos = 0
        seconds = 0
        for df in files.values():
            if "length" in df.columns:
                videos += len(df)
                seconds += df["length"].sum()
        hours = seconds / 3600
        channel_stats[channel_name] = {"videos": videos, "seconds": seconds, "hours": hours}
        total_videos += videos
        total_seconds += seconds

    total_hours = total_seconds / 3600

    # Determine column widths for formatted output
    max_channel_name = max(len(name) for name in channel_stats)
    channel_videos_length = len(str(total_videos))
    channel_hours_length = len(f"{total_hours:.2f}")
    # max_channel_videos = max(len(str(stats["videos"])) for stats in channel_stats.values())
    # max_channel_hours = max(len(f"{stats['hours']:.2f}") for stats in channel_stats.values())

    # Print statistics for each channel sorted alphabetically (case-insensitive)
    # Print header with column names
    videos_header = channel_videos_length + len(" videos")
    hours_header = channel_hours_length + len(" hours")
    print(f"\n{'Channel':<{max_channel_name}} | {'Videos':>{videos_header}} | {'Duration':>{hours_header}} | {'Percent':>7}")
    print(f"{'-' * max_channel_name} | {'-' * videos_header} | {'-' * hours_header} | {'-' * 7}")
    for channel_name in sorted(channel_stats.keys(), key=str.lower):
        stats = channel_stats[channel_name]
        percentage = (stats["hours"] / total_hours * 100) if total_hours > 0 else 0
        print(f"{channel_name:<{max_channel_name}} | " f"{stats['videos']:>{channel_videos_length}} videos | " f"{stats['hours']:>{channel_hours_length}.2f} hours | " f"{percentage:>6.2f}%")

    print(f"{'-' * max_channel_name} | {'-' * videos_header} | {'-' * hours_header} | {'-' * 7}")
    print(f"{'Total':<{max_channel_name}} | {total_videos:>{channel_videos_length}} videos | {total_hours:>{channel_hours_length}.2f} hours | 100.00%")

    return total_videos, total_hours


def count_videos(data_dir: str = "data") -> None:
    """Read CSV files and calculate video statistics.

    Args:
        data_dir (Path): The root data directory containing channel subdirectories.
    """
    all_csv_data = read_channel_csv_files(data_dir)
    calculate_channel_statistics(all_csv_data)


def count_files(data_dir: str = "data") -> None:
    """Count and compare m4a and SRT files in data directories.

    Args:
        data_dir (Path): The root data directory to search for files.
    """
    data_dir = Path(data_dir)
    dir_counts = {}
    total_m4a = 0
    total_srt = 0

    # Process only leaf directories (no subdirectories)
    for dirpath, dirs, filenames in os.walk(data_dir):
        if not dirs:
            m4a_files = [f for f in filenames if f.endswith(".m4a")]
            srt_files = [f for f in filenames if f.endswith(".srt")]

            m4a_count = len(m4a_files)
            srt_count = len(srt_files)

            total_m4a += m4a_count
            total_srt += srt_count

            dir_name = Path(dirpath).name
            dir_counts[dir_name] = {"m4a": m4a_count, "srt": srt_count}

    max_dir_name = max(len(name) for name in dir_counts) if dir_counts else 0
    max_m4a_count = max(len(str(stats["m4a"])) for stats in dir_counts.values()) if dir_counts else 0
    max_srt_count = max(len(str(stats["srt"])) for stats in dir_counts.values()) if dir_counts else 0

    # Print header with column names
    m4a_header = max(max_m4a_count, len("m4a files"))
    srt_header = max(max_srt_count, len("SRT files"))
    match_header = len("Same")
    print(f"\n{'Directory':<{max_dir_name}} | {'m4a files':>{m4a_header}} | {'SRT files':>{srt_header}} | {'Same':>{match_header}}")
    print(f"{'-' * max_dir_name} | {'-' * m4a_header} | {'-' * srt_header} | {'-' * match_header}")

    # Print statistics for each directory sorted alphabetically
    for dir_name in sorted(dir_counts.keys(), key=str.lower):
        stats = dir_counts[dir_name]
        match_status = "Yes" if stats["m4a"] == stats["srt"] else "No"
        print(f"{dir_name:<{max_dir_name}} | {stats['m4a']:>{m4a_header}} | {stats['srt']:>{srt_header}} | {match_status:>{match_header}}")

    print(f"\nTotal: {total_m4a} m4a files, {total_srt} SRT files, Same: {'Yes' if total_m4a == total_srt else 'No'}")


if __name__ == "__main__":
    count_videos(data_dir="data_en")
    count_files(data_dir="data_en")
