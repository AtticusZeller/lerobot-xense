# Guided recording

Guided mode appends episodes to the local LeRobot dataset used to generate a reviewed [collection plan](https://github.com/AtticusZeller/lerobot-check/blob/b7060b102e50898b6929346dfdfe75c36acda532/docs/collection-plan.md). The recorder uses its own head camera; close any separate camera preview before starting. To record as usual, omit `--collection_plan`.

## Start a session

1. Export the reviewed plan from the viewer's COLLECTION tab. Copy the JSON to the recording machine and use a writable copy there. Do not regenerate the original plan or open another recorder against the same local dataset during the session.
2. Use the matching recording recipe and a complete local copy of the dataset. The repo ID, baseline revision, head image feature, camera serial and native frame size must match. As in `lerobot-check`, the baseline is the single cached revision under `.cache/huggingface/trees/`, or `local` when there is no single cached revision. Keep the matching tree with a verified dataset copy; do not retag an unrelated dataset or resize a mismatched camera feed.
3. Start from the recording machine's desktop, with an OpenCV GUI backend and the recorder's usual keyboard support. To check the GUI build in the recorder environment, run `python -c 'import cv2; print(cv2.currentUIFramework())'`. If the result is empty, enable the GUI OpenCV build (the headless wheel also provides `cv2`). Keep the desktop session's `DISPLAY` and authorization when launching the process.

For the Flexiv bench that captured `pack_iphone_box_1119`, use its bench-specific `pack_iphone-taccap.yaml`; do not substitute another bench's recipe. With the correct recipe and local dataset paths:

```bash
lerobot-record --config_path="$RECIPE" \
  --resume=true --dataset.root="$DATASET" \
  --dataset.push_to_hub=false --dataset.video_encoding_batch_size=1 \
  --display_data=false --collection_plan="$PLAN"
```

The recorder opens the guide before the first take and before every subsequent pending round. Reset does not time out. The guide shows the round and progress, numbered target outlines and arrows, and, when available, observed outlines and mask overlap (IoU). A green message and a spoken prompt (when `play_sounds` is on) indicate all targets reached the threshold. The score checks only the visible outline: confirm object identity and orientation yourself. A target without an outline shows a point and arrow and always needs manual judgment.

| Key         | During reset                                                               | During recording                                           |
| ----------- | -------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Enter       | Confirm in the guide window, with or without passing scores.               | No action.                                                 |
| Right arrow | Does not confirm.                                                          | End and save the take early.                               |
| Left arrow  | Does not discard a previously saved take.                                  | Discard and return to the same round.                      |
| Escape      | Stop without recording the pending round. Closing the guide does the same. | End and save the current take if it has frames, then stop. |

On confirmation, the guide stops accepting frames for inference; recording starts after the active inference finishes and the guide closes. Successful saves close the episode data and metadata Parquet shards before updating the plan with the actual zero-based episode index. The session stops when all pending rounds are done or the requested episode cap is reached. If a take saved but the plan write failed, check the recorded plan ID, round ID and episode index in the log before using `lerobot-check collection-plan mark-round` to repair progress. Do not guess the mapping from the next episode number.

## Optional automatic alignment

In the existing recorder environment, install the compatible `lerobot-check` build without its offline analysis dependencies, and install the model dependency from its `mask-template` extra. Using `--no-deps` for `lerobot-check` avoids replacing the recorder's LeRobot or GUI OpenCV with the offline package's dependencies.

```bash
uv pip install --no-deps \
  'lerobot-check @ git+https://github.com/AtticusZeller/lerobot-check.git@b7060b102e50898b6929346dfdfe75c36acda532'
uv pip install 'transformers>=5.24.1,<6'
hf download facebook/sam2.1-hiera-tiny config.json preprocessor_config.json \
  processor_config.json model.safetensors \
  --revision de431c4043854a71d8101e17995dfe596bf101a5
python -c 'from lerobot_check.alignment import AlignmentEvaluator; AlignmentEvaluator.from_files(device="cuda")'
```

The recorder does not download weights. A missing dependency, unavailable CUDA or failed inference leaves the target overlay and manual confirmation available. Omit `--collection_alignment_thresholds` to use the default threshold of 0.60 for every object. To override it, pass `--collection_alignment_thresholds=/path/thresholds.json`. An optional `per_object` map overrides `min_iou` for the named targets; files with just `version` and `min_iou` still work.

```json
{ "version": 1, "min_iou": 0.6, "per_object": { "box_lid": 0.5 } }
```

The guide compares each score to the threshold for that target. A score up to five percentage points below its threshold asks for visual confirmation without telling the operator to move the object. The example override is illustrative; keep the default until bench measurements support a change.

## Verify on the Flexiv bench

Only an authorized operator on site should run this check. Use the correct bench recipe, a local test copy and a copy of the reviewed plan; keep automatic Hub upload off.

1. Before confirming the first round, check that the head view is upright and live, the target numbers and outlines match the reviewed plan, and moving the objects updates the observed contours and IoU. Check that the passing state and audio occur when all targets align. A manual-only round must not claim automatic success.
2. Confirm with Enter. The window must close and stay closed during capture. Press Left arrow to discard a take, then verify the same round reopens and is still pending. Confirm again, record and save it, then verify the next round appears and the plan records the saved episode index.
3. Confirm the next round even without passing alignment. During capture, press Escape after recording frames. Check that the recorder saves that take, marks its round with a distinct episode index, and exits. On a subsequent reset, Escape must exit without consuming the pending round. Check both saved episodes in the local viewer (head video, data and metadata).
4. Return the updated plan JSON to the review workstation and merge it with `lerobot-check collection-plan import-progress --plan "$ORIGINAL" --returned "$RETURNED"`. **No dataset is pushed automatically.** After checking the dataset, upload it separately with an authorized writer. The default Hub login on `xense-robot-flexiv` cannot write `Xense/*`; use the authorized account on `xense-robot-elite` following the bench credential procedure, without changing the Flexiv login.
