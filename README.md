# autoset

`autoset` is a script-first pipeline for turning raw anime source videos into short, tagged, captioned, train/val datasets with split-local `ltx.jsonl` manifests.

The repo is built around six main stages:

1. Download source videos.
2. Manually review and keep only good source material.
3. Extract short clips while avoiding scene boundaries.
4. Tag clips with wd-tagger.
5. Build source-disjoint `train/` and `val/` splits.
6. Caption each split and export LTX-ready JSONL manifests.

## What "LTX-ready" means here

In this repo, an LTX-ready dataset is a subset directory with this shape:

```text
data/clips_subset/
  train/
    ...
    captions.jsonl
    ltx.jsonl
  val/
    ...
    captions.jsonl
    ltx.jsonl
```

Each `ltx.jsonl` row contains:

```json
{"caption": "A character sprints across a city rooftop at dusk while the camera tracks alongside them and traffic hums below.", "media_path": "clip_1.mp4"}
```

`media_path` is relative to the split directory, so `train/ltx.jsonl` points at files inside `train/`, and `val/ltx.jsonl` points at files inside `val/`.

Optional helper:

```bash
python3 anime_clip_buckets.py \
  --input-dir data/clips_subset/train \
  --resolution 512
```

This prints up to 30 semicolon-delimited `widthxheightxframes` buckets derived from the clips in that directory.

## Requirements

- Python 3.10 or newer
- FFmpeg on `PATH`, with both `ffmpeg` and `ffprobe` available
- A Python build with `tkinter` if you want to use the review UI
- Enough disk space for raw downloads plus extracted clips
- Enough RAM / compute for the caption model you choose

Notes:

- `anime_clip_extract.py` supports `--device auto|cpu|cuda|mps`.
- `anime_clip_caption.py` supports `--device cuda|cpu` and defaults to `cuda`.
- If you are on Apple Silicon, extraction can use `--device mps`, but captioning must use `--device cpu`.
- Captioning downloads a Hugging Face multimodal model. If the repo you choose is gated, make sure your Hugging Face access is set up before that step.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Quick sanity checks:

```bash
ffmpeg -version
ffprobe -version
python -c "import tkinter; print('tkinter ok')"
```

## Recommended workspace layout

You can use any paths you want, but using explicit directories makes reruns easier to reason about:

```text
data/
  downloads/
  downloads_keep/
  downloads_reject/
  clips/
  clips_subset/
```

The commands below use that layout.

## End-to-end guide

### 1. Download raw source videos

Use `sakuga_download.py` to fetch videos from Sakugabooru:

```bash
python sakuga_download.py \
  --query "yutaka_nakamura" \
  --count 500 \
  --output-dir data/downloads
```

What it does:

- Fetches matching Sakugabooru posts.
- Downloads only supported video posts (`mp4`, `webm`).
- Resumes partial `.part` downloads when possible.
- Skips files that are already fully downloaded.
- Prints a final summary like `requested=... matched=... downloaded=... skipped=... failed=...`.

Tips:

- If you do not include an explicit `order:...` token in `--query`, the downloader automatically adds `order:id_desc`.
- Rerunning the same command against the same `--output-dir` is fine. Existing completed files are skipped, and partial files are resumed.

### 2. Review the raw videos and keep only usable source material

Use `anime_video_review.py` on the download directory:

```bash
python anime_video_review.py --input-dir data/downloads
```

By default this creates:

- `data/downloads_keep/` for accepted videos
- `data/downloads_reject/` for rejected videos
- `data/downloads_review.jsonl` as the resumable review log

UI controls:

- `Right`: move to next video
- `Left`: move to previous video
- `A`: accept current video immediately
- `R`: reject current video immediately

Important behavior:

- When you first land on a video, it is still `pending`.
- The reviewer loops a 2-second middle preview and preloads nearby previews to keep navigation responsive.
- If you move away from a seen `pending` video without rejecting it, the app auto-accepts it.
- Accepted files are moved into `*_keep/`.
- Rejected files are moved into `*_reject/`.
- The review log is append-only and lets the session resume cleanly after restarts.

Useful option:

```bash
python anime_video_review.py --input-dir data/downloads --watch
```

`--watch` keeps polling the input directory for newly completed downloads, which is handy if you want to review while a long download job is still running.

### 3. Extract short clips from accepted source videos

Run clip extraction on the reviewed keep set:

```bash
python3 anime_clip_extract.py \
  --input-dir data/downloads_keep \
  --output-dir data/clips \
  --min-duration 1.0 \
  --max-duration 5.0 \
  --boundary-threshold 0.25 \
  --boundary-padding 0.75 \
  --min-gap 0.25 \
  --device cpu \
  --seed 0
```

What it does:

- Uses TransNetV2 in a recall-biased mode to find scene boundaries.
- Avoids padded boundary regions, plus an implicit one-frame guard on both sides of each detected cut.
- Tiles the remaining safe spans into 1s-5s clips.
- Preserves video and optional audio in the extracted clip.
- Writes `data/clips/manifest.jsonl` as the source-to-clip manifest.

Expected output shape:

```text
data/clips/
  manifest.jsonl
  episode_01/
    clip_1_0_2200.mp4
    clip_2_2450_5100.mp4
  episode_02/
    clip_1_500_4100.mp4
```

`manifest.jsonl` rows include:

- `source_path`
- `clip_path`
- `start`
- `end`
- `duration`
- `boundary_threshold`
- `boundary_padding`
- `boundary_mode`
- `seed`

Notes:

- Extraction is deterministic for a given `--seed` and source path.
- The default boundary settings are intentionally recall-biased and may discard more usable footage to reduce missed scene cuts.
- The script overwrites `manifest.jsonl`, but it does not clean out old clip files for you. If you change extraction settings, using a fresh `--output-dir` is the safest choice.
- The command exits non-zero if no clips are created or if any source video fails.

### 4. Tag the extracted clips

Run wd-tagger on the extracted clip directory:

```bash
python3 anime_clip_tag.py \
  --input-dir data/clips \
  --output-file data/clips/tags.jsonl \
  --model-repo SmilingWolf/wd-swinv2-tagger-v3 \
  --general-threshold 0.35 \
  --character-threshold 0.9 \
  --batch-size 8
```

What it does:

- Samples one representative frame from the middle of each clip.
- Runs wd-tagger on that frame.
- Writes one JSONL row per clip to `tags.jsonl`.

Important output fields:

- `clip_path`
- `clip_relpath`
- `duration`
- `frame_time`
- `rating`
- `general_tags`
- `general_tag_scores`
- `character_tags`
- `character_tag_scores`
- `all_tags`
- `status`
- `error`

Success and failure behavior:

- Successful rows have `status="tagged"`.
- Failed rows stay in the manifest with `status="failed"` and an `error` message.
- The script exits non-zero if any clip fails or if zero clips are tagged successfully.

### 5. Build source-disjoint train/val splits

Use the tag manifest to copy clips into `train/` and `val/`:

```bash
python3 anime_clip_subset.py \
  --input-dir data/clips \
  --tags-file data/clips/tags.jsonl \
  --output-dir data/clips_subset \
  --fraction 0.10
```

Or choose exact validation and train sizes:

```bash
python3 anime_clip_subset.py \
  --input-dir data/clips \
  --tags-file data/clips/tags.jsonl \
  --output-dir data/clips_subset \
  --count 250 \
  --train-count 1000
```

What it does:

- Ignores rows whose tag `status` is not `tagged`.
- Uses `manifest.jsonl` from extraction when available so clips from the same source video stay together.
- Builds source-disjoint train and validation splits.
- Keeps both splits diverse and balanced with tag-aware selection heuristics.
- Copies selected files into `train/` and `val/`.
- Writes split-local `tags.jsonl` manifests.

Expected output shape:

```text
data/clips_subset/
  train/
    tags.jsonl
    episode_01/
      clip_1_0_2200.mp4
  val/
    tags.jsonl
    episode_02/
      clip_1_500_4100.mp4
```

Important constraints:

- You need at least two distinct source videos to make source-disjoint `train` and `val`.
- Omitting `--train-count` keeps the current behavior and uses all eligible remaining clips for `train/`.
- `--output-dir` must be empty of video files and existing `tags.jsonl` manifests.
- The split manifests add fields like `split`, `source_id`, `selection_rank`, `new_tags_added`, and `balance_penalty`.

Benchmark helper:

```bash
python3 benchmark_anime_clip_subset.py --clips 100 --fraction 0.10 --train-count 30
python3 benchmark_anime_clip_subset.py --clips 3500 --fraction 0.10 --train-count 1000
```

If you want to time a real run instead of a synthetic one, redirect the verbose selection output to a file:

```bash
time python3 anime_clip_subset.py \
  --input-dir data/clips \
  --tags-file data/clips/tags.jsonl \
  --output-dir data/clips_subset \
  --fraction 0.10 \
  --train-count 1000 \
  > /tmp/anime_clip_subset.log
```

### 6. Caption each split

Caption `train/` and `val/` separately. The LTX export step expects a `captions.jsonl` inside each split directory.

Example with a Gemma 4 multimodal model:

```bash
python anime_clip_caption.py \
  --input-dir data/clips_subset/train \
  --model-repo google/gemma-4-E4B-it \
  --cache-dir /runpod-volume/hf-cache \
  --device cuda
```

```bash
python anime_clip_caption.py \
  --input-dir data/clips_subset/val \
  --model-repo google/gemma-4-E4B-it \
  --cache-dir /runpod-volume/hf-cache \
  --device cuda
```

If you do not have CUDA, use CPU:

```bash
python3 anime_clip_caption.py \
  --input-dir data/clips_subset/train \
  --model-repo google/gemma-4-E2B-it \
  --device cpu
```

What it does:

- Loads the chosen Hugging Face multimodal model.
- If `--cache-dir` is set, stores downloaded Hugging Face model files there so they can persist across pod restarts when backed by network storage.
- Loads all frames with `--sample-all-frames`, or by default samples at least half the source frames when metadata is available, with `--num-frames` acting as a floor.
- Uses `tags.jsonl` in the same directory as helper hints when present.
- Includes audio only when the model supports audio and the clip has an audio track.
- Writes `captions.jsonl`.

Important output fields:

- `clip_path`
- `duration`
- `has_audio_track`
- `used_audio`
- `audio_skip_reason`
- `scene_overview`
- `visual_details`
- `dialogue`
- `other_sounds`
- `caption`
- `status`
- `error`

Caption format:

- Successful captions are written as a single plain paragraph of continuous text in strict chronological order.
- Captions do not include section headers such as `SCENE_OVERVIEW`, `VISUAL_DETAILS`, `DIALOGUE`, `OTHER_SOUNDS`, `VISUAL`, or `AUDIO`.
- Captions are intended to be exhaustive, evidence-only descriptions for video-generation training data and should omit details that are not clearly visible or audible.
- When audio is used, audible dialogue and other sounds are blended naturally into the same paragraph at the point where they occur.
- The compatibility fields `scene_overview`, `visual_details`, `dialogue`, and `other_sounds` remain in the manifest but are left `null` for newly generated captions.

Failure behavior:

- Failed rows are still written with `status="failed"` and `error`.
- The script exits non-zero if any clip fails or if zero clips are captioned successfully.

### 7. Export LTX-ready manifests

Once both split directories have successful `captions.jsonl` files, run:

```bash
python anime_clip_ltx.py --input-dir data/clips_subset
```

This creates:

- `data/clips_subset/train/ltx.jsonl`
- `data/clips_subset/val/ltx.jsonl`

What it validates:

- `train/` and `val/` both exist.
- Each split has videos.
- Each split has a caption manifest.
- Every video in each split has exactly one `status="captioned"` row.
- Every caption row points at a clip inside the same split.

If any clip is missing a successful caption row, export fails instead of silently producing a partial dataset.

### 8. Inspect the finished captions

Use the stats helper on the final manifests:

```bash
python anime_dataset_word_stats.py \
  data/clips_subset/train/ltx.jsonl \
  data/clips_subset/val/ltx.jsonl
```

This prints:

- number of manifests
- number of caption rows
- unique word count
- unique bigram count
- top words
- top bigrams

This is useful for quickly spotting repetitive captioning or bad prompt drift before training.

## Minimal copy-paste runbook

If you want the shortest possible path from zero to final manifests:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python sakuga_download.py \
  --query "yutaka_nakamura" \
  --count 500 \
  --output-dir data/downloads

python anime_video_review.py --input-dir data/downloads

python anime_clip_extract.py \
  --input-dir data/downloads_keep \
  --output-dir data/clips \
  --device cpu

python anime_clip_tag.py --input-dir data/clips

python anime_clip_subset.py \
  --input-dir data/clips \
  --output-dir data/clips_subset \
  --fraction 0.10 \
  --train-count 1000

python anime_clip_caption.py \
  --input-dir data/clips_subset/train \
  --model-repo google/gemma-4-E4B-it \
  --device cuda

python anime_clip_caption.py \
  --input-dir data/clips_subset/val \
  --model-repo google/gemma-4-E4B-it \
  --device cuda

python anime_clip_ltx.py --input-dir data/clips_subset
```

Final artifacts:

- `data/clips_subset/train/ltx.jsonl`
- `data/clips_subset/val/ltx.jsonl`

## Rerun and recovery guide

When a step fails or you want to rerun part of the pipeline:

- Download: safe to rerun against the same output directory. Complete files are skipped and partial files are resumed.
- Review: safe to reopen. The JSONL review log restores prior decisions.
- Extract: best rerun into a fresh output directory if you change extraction parameters.
- Tag: rerunning rewrites the output manifest.
- Subset: rerunning requires an empty output directory.
- Caption: rerunning rewrites the output manifest for that split.
- LTX export: rerunning rewrites `ltx.jsonl`.

## Common pitfalls

- Running captioning on `data/clips_subset/` instead of on `data/clips_subset/train` and `data/clips_subset/val` separately.
- Forgetting to caption both splits before running `anime_clip_ltx.py`.
- Trying to create a split when all usable clips come from only one source video.
- Reusing an old extraction output directory after changing clip extraction settings.
- Forgetting that review navigation auto-accepts a seen pending video unless you explicitly reject it.

## Script reference

- `sakuga_download.py`: fetch raw Sakugabooru video posts
- `anime_video_review.py`: manually accept/reject source videos in a Tk UI
- `anime_clip_extract.py`: extract short safe clips from reviewed sources
- `anime_clip_tag.py`: tag clips from a representative frame
- `anime_clip_subset.py`: create balanced, source-disjoint `train/` and `val/`
- `anime_clip_caption.py`: caption clips with a Hugging Face multimodal model
- `anime_clip_ltx.py`: build final LTX manifests
- `anime_dataset_word_stats.py`: inspect final caption vocabulary
