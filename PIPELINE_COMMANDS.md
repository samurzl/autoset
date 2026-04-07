# Pipeline Command Examples

These examples assume this path layout:

```text
data/
  downloads/
  downloads_keep/
  downloads_reject/
  clips/
  clips_subset/
```

Every script below also accepts `-h` and `--help` to show its built-in help text and exit.

## 1. Download source videos

```bash
python3 sakuga_download.py \
  --query "order:score -fighting" \
  --count 500 \
  --output-dir data/downloads \
  --concurrency 8
```

Options:

- `--query QUERY` (required): Raw Sakugabooru tag query. If the query does not already contain an `order:` token, the script appends `order:id_desc`.
- `--count COUNT` (required): Maximum number of videos to download. Must be a positive integer.
- `--output-dir OUTPUT_DIR`: Base output directory. Default: `downloads/<query-slug>/`.
- `--concurrency CONCURRENCY`: Number of concurrent downloads. Default: `min(count, 8)`. Must be a positive integer.

## 2. Review downloaded videos into keep and reject sets

```bash
python3 anime_video_review.py \
  --input-dir data/downloads
```

Review while downloads are still being added:

```bash
python3 anime_video_review.py \
  --input-dir data/downloads \
  --watch
```

Options:

- `--input-dir INPUT_DIR` (required): Directory containing downloaded videos.
- `--keep-dir KEEP_DIR`: Directory for accepted videos. Default: sibling directory named `<input-dir>_keep`.
- `--reject-dir REJECT_DIR`: Directory for rejected videos. Default: sibling directory named `<input-dir>_reject`.
- `--review-log REVIEW_LOG`: JSONL log used to resume decisions and store preview metadata. Default: sibling file named `<input-dir>_review.jsonl`.
- `--preview-duration PREVIEW_DURATION`: Loop preview duration in seconds. Default: `2.0`. Must be greater than `0`.
- `--watch`: Keep polling `--input-dir` for newly completed downloads.

## 3. Extract short clips from the accepted videos

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

Options:

- `--input-dir INPUT_DIR` (required): Directory containing source videos to cut into clips.
- `--output-dir OUTPUT_DIR`: Directory where clips are written. Default: `clips/`.
- `--min-duration MIN_DURATION`: Minimum clip duration in seconds. Default: `1.0`. Must be greater than `0`.
- `--max-duration MAX_DURATION`: Maximum clip duration in seconds. Default: `5.0`. Must be greater than `0` and at least `--min-duration`.
- `--boundary-threshold BOUNDARY_THRESHOLD`: Scene-boundary probability threshold used by TransNetV2. Lower values are more recall-biased and may exclude more ambiguous cuts. Default: `0.25`.
- `--boundary-padding BOUNDARY_PADDING`: Extra seconds to avoid on both sides of detected boundaries, on top of an implicit one-frame guard per side. Larger values are more recall-biased. Default: `0.75`. Must be non-negative.
- `--min-gap MIN_GAP`: Gap in seconds left between extracted clips. Default: `0.25`. Must be non-negative.
- `--device {auto,cpu,cuda,mps}`: Device used for TransNetV2 inference. Default: `cpu`.
- `--seed SEED`: Global seed used for deterministic per-video clip sampling. Default: `0`.

The extractor intentionally favors recall over yield, so it may drop more usable footage to reduce missed scene cuts.

## 4. Tag the extracted clips

```bash
python3 anime_clip_tag.py \
  --input-dir data/clips \
  --output-file data/clips/tags.jsonl \
  --model-repo SmilingWolf/wd-swinv2-tagger-v3 \
  --general-threshold 0.35 \
  --character-threshold 0.9 \
  --batch-size 8
```

Options:

- `--input-dir INPUT_DIR` (required): Directory containing extracted clips.
- `--output-file OUTPUT_FILE`: JSONL output manifest path. Default: `<input-dir>/tags.jsonl`.
- `--model-repo MODEL_REPO`: Hugging Face repo for wd-tagger. Default: `SmilingWolf/wd-swinv2-tagger-v3`.
- `--general-threshold GENERAL_THRESHOLD`: Score threshold for general tags. Default: `0.35`. Must be between `0` and `1`.
- `--character-threshold CHARACTER_THRESHOLD`: Score threshold for character tags. Default: `0.9`. Must be between `0` and `1`.
- `--batch-size BATCH_SIZE`: Inference batch size. Default: `8`. Must be a positive integer.

## 5. Build source-disjoint train and validation splits

Fraction-based validation split:

```bash
python3 anime_clip_subset.py \
  --input-dir data/clips \
  --tags-file data/clips/tags.jsonl \
  --output-dir data/clips_subset \
  --fraction 0.10
```

Exact-size validation split:

```bash
python3 anime_clip_subset.py \
  --input-dir data/clips \
  --tags-file data/clips/tags.jsonl \
  --output-dir data/clips_subset \
  --count 250
```

Exact-size validation and train splits:

```bash
python3 anime_clip_subset.py \
  --input-dir data/clips \
  --tags-file data/clips/tags.jsonl \
  --output-dir data/clips_subset \
  --count 250 \
  --train-count 1000
```

Options:

- `--input-dir INPUT_DIR` (required): Directory containing the tagged clips.
- `--tags-file TAGS_FILE`: JSONL tag manifest path. Default: `<input-dir>/tags.jsonl`.
- `--output-dir OUTPUT_DIR`: Output directory for `train/` and `val/`. Default: sibling directory named `<input-dir>_subset`.
- `--count COUNT`: Exact number of successfully tagged clips to place in `val/`. Must be a positive integer.
- `--fraction FRACTION`: Fraction of successfully tagged clips to place in `val/`. Must be greater than `0` and less than or equal to `1`.
- `--train-count TRAIN_COUNT`: Exact number of successfully tagged clips to place in `train/`. Default: all eligible remaining clips.

Exactly one of `--count` or `--fraction` is required. `--output-dir` must be different from `--input-dir` and must not be inside `--input-dir`. Both splits stay source-disjoint and are selected to stay diverse and balanced.

## 6. Caption the train split

```bash
python3 anime_clip_caption.py \
  --input-dir data/clips_subset/train \
  --model-repo google/gemma-4-E2B-it \
  --device cpu
```

## 7. Caption the validation split

```bash
python3 anime_clip_caption.py \
  --input-dir data/clips_subset/val \
  --model-repo google/gemma-4-E2B-it \
  --device cuda
```

CPU example:

```bash
python3 anime_clip_caption.py \
  --input-dir data/clips_subset/train \
  --model-repo google/gemma-4-E2B-it \
  --device cpu
```

Options for `anime_clip_caption.py`:

- `--input-dir INPUT_DIR` (required): Directory containing clips for one split.
- `--model-repo MODEL_REPO` (required): Hugging Face repo for the Gemma 4 multimodal model.
- `--output-file OUTPUT_FILE`: JSONL output manifest path. Default: `<input-dir>/captions.jsonl`.
- `--tags-file TAGS_FILE`: Optional JSONL tag manifest used as helper hints. Default: `<input-dir>/tags.jsonl` if that file exists.
- `--device {cuda,cpu}`: Device used for inference. Default: `cuda`.
- `--dtype {auto,float16,bfloat16,float32}`: Model dtype hint. Default: `auto`.
- `--num-frames NUM_FRAMES`: Number of sampled frames per clip. Default: `12`. Must be a positive integer.
- `--max-new-tokens MAX_NEW_TOKENS`: Maximum number of generated tokens per clip. Default: `768`. Must be a positive integer.

## 8. Export LTX-ready manifests

```bash
python3 anime_clip_ltx.py \
  --input-dir data/clips_subset
```

Options:

- `--input-dir INPUT_DIR` (required): Subset root containing `train/` and `val/`.
- `--captions-filename CAPTIONS_FILENAME`: Caption manifest filename expected inside each split directory. Default: `captions.jsonl`.
- `--output-filename OUTPUT_FILENAME`: Output filename written inside each split directory. Default: `ltx.jsonl`.

## 9. Inspect the final manifests for word and bigram repetition

```bash
python3 anime_dataset_word_stats.py \
  --top-n 25 \
  data/clips_subset/train/ltx.jsonl \
  data/clips_subset/val/ltx.jsonl
```

Arguments and options:

- `manifest_paths` (required positional): One or more JSONL manifests containing a `caption` field.
- `--top-n TOP_N`: Number of top words and top bigrams to print. Default: `25`. Must be a positive integer.

## Optional helper: derive dataset bucket strings

```bash
python3 anime_clip_buckets.py \
  --input-dir data/clips_subset/train \
  --resolution 512
```

Options:

- `--input-dir INPUT_DIR` (required): Directory containing clips to scan recursively.
- `--resolution RESOLUTION` (required): Square-root area target `N`, so projected bucket areas approximate `N^2`.
- `--max-buckets MAX_BUCKETS`: Maximum number of buckets to print. Default: `30`.
