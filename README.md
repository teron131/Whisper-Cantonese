---
configs:
  - config_name: all
    data_files: "*/*.tar"
    default: true
  - config_name: ahju
    data_files: ahju/*.tar
  - config_name: bboblackboxoffice
    data_files: bboblackboxoffice/*.tar
  - config_name: dangerousperson2.0
    data_files: dangerousperson2.0/*.tar
  - config_name: greenbeanmediaofficial
    data_files: greenbeanmediaofficial/*.tar
  - config_name: hieggo1001
    data_files: hieggo1001/*.tar
  - config_name: hkcrime
    data_files: hkcrime/*.tar
  - config_name: jerson8964
    data_files: jerson8964/*.tar
  - config_name: maviskuku
    data_files: maviskuku/*.tar
  - config_name: mingjai14
    data_files: mingjai14/*.tar
  - config_name: mm.millmilk
    data_files: mm.millmilk/*.tar
  - config_name: mpweekly
    data_files: mpweekly/*.tar
  - config_name: pinkytalks
    data_files: pinkytalks/*.tar
  - config_name: pricehongkongofficial
    data_files: pricehongkongofficial/*.tar
  - config_name: sunchannelhk
    data_files: sunchannelhk/*.tar
  - config_name: thedoshow0909
    data_files: thedoshow0909/*.tar
  - config_name: unwire
    data_files: unwire/*.tar
license: mit
task_categories:
  - automatic-speech-recognition
language:
  - zh
  - yue
size_categories:
  - 100K<n<1M
---

# Cantonese Audio Dataset from YouTube

This dataset contains Cantonese audio segments extracted from various YouTube channels, along with corresponding transcription metadata. The data is intended for training automatic speech recognition (ASR) models.

## Data Source and Processing

The data was obtained through the following process:

1.  **Download:** Audio (`.m4a`) and available Cantonese subtitles (`.srt` for `zh-TW`, `zh-HK`, `zh-Hant`) were downloaded from selected YouTube channels. This raw data, along with video metadata (`metadata.csv`), is stored initially in a `data/{channel_id}/` directory structure.
2.  **Segmentation:** The raw audio files were segmented based on the timing information in the `.srt` files.
    - Audio files are splitted by SRT segments and then combined to a maximum duration less than but close to 30 seconds per group for Whisper.
    - The corresponding audio portions for each group are extracted using `ffmpeg` and saved as `.mp3` files at a 16000 Hz sample rate.
    - Metadata for each segment, including channel/video info and the text/timing of subtitles within the segment, is saved in a corresponding `.json` file.

## Intermediate Dataset Structure (`dataset` directory)

Before being packaged into TAR archives for Hugging Face, the segmented data resides in the `dataset` directory with the following structure:

```
dataset/
└── {channel_id}/             # Directory named after the YouTube channel ID
    └── {video_id}/           # Directory named after the YouTube video ID
        ├── {video_id}_{group_name}.mp3  # Segmented audio file
        ├── {video_id}_{group_name}.json # Corresponding metadata file
        └── ...
```

- **`{channel_id}`:** The ID of the YouTube channel (e.g., `greenbeanmediaofficial`).
- **`{video_id}`:** The unique identifier for the YouTube video.
- **`{group_name}`:** Represents the subtitles included in the segment. It's either the index of the first subtitle (e.g., `1`) if the group contains only one, or a range indicating the first and last subtitle indices (e.g., `1-5`) if the group contains multiple subtitles.

## Dataset Summary

The dataset comprises audio from the following channels:

```
Channel                |      Videos |     Duration | Percent
---------------------- | ----------- | ------------ | -------
AhJu                   |  132 videos |  28.81 hours |   1.56%
BBOBlackboxoffice      |  122 videos |  32.66 hours |   1.76%
DangerousPerson2.0     |  114 videos |  70.53 hours |   3.81%
greenbeanmediaofficial |  594 videos | 179.97 hours |   9.71%
hieggo1001             | 1251 videos | 279.30 hours |  15.07%
hkcrime                |   99 videos |  35.06 hours |   1.89%
JERSON8964             |  500 videos |  97.60 hours |   5.27%
maviskuku              |  165 videos |  29.21 hours |   1.58%
mingjai14              |  158 videos |  43.85 hours |   2.37%
mm.millmilk            |  958 videos | 271.25 hours |  14.64%
MPWeekly               | 1119 videos | 156.45 hours |   8.44%
pinkytalks             |  125 videos |  20.72 hours |   1.12%
pricehongkongofficial  |  959 videos | 131.94 hours |   7.12%
SunChannelHK           | 1160 videos | 409.18 hours |  22.08%
TheDoShow0909          |   23 videos |  17.78 hours |   0.96%
unwire                 |  345 videos |  48.53 hours |   2.62%
---------------------- | ----------- | ------------ | -------
Total                  | 7824 videos | 1852.83 hours| 100.00%
```

## Loading the Data

You can load the data using the Hugging Face `datasets` library:

```python
import os

from datasets import load_dataset

ds = load_dataset(
    "OrcinusOrca/YouTube-Cantonese",
    "all",  # or channel_id as config
    split="train",
    streaming=False,  # or True
    num_proc=os.cpu_count(),
)
```

```python
>>> ds
Dataset({
    features: ['mp3', 'json', '__key__', '__url__'],
    num_rows: 216009
})

>>> ds.features
{'mp3': Audio(sampling_rate=None, mono=True, decode=True, id=None),
 'json': {'caption': Value(dtype='string', id=None),
  'caption_code': Value(dtype='string', id=None),
  'channel_id': Value(dtype='string', id=None),
  'length': Value(dtype='string', id=None),
  'segments': [{'end_time': Value(dtype='string', id=None),
    'segment_index': Value(dtype='int64', id=None),
    'start_time': Value(dtype='string', id=None),
    'text': Value(dtype='string', id=None)}],
  'title': Value(dtype='string', id=None),
  'url': Value(dtype='string', id=None),
  'video_id': Value(dtype='string', id=None)},
 '__key__': Value(dtype='string', id=None),
 '__url__': Value(dtype='string', id=None)}
```
