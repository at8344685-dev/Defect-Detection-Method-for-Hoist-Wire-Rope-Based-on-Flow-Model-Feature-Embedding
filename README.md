# Defect-Detection-Method-for-Hoist-Wire-Rope-Based-on-Flow-Model-Feature-Embedding
Implementation of the paper "Defect Detection Method for Hoist Wire Rope Based on Flow Model Feature Embedding"

The system consists of four modules:

- MEPEM: Mine Environment Perception Enhancement Module
  - VFE: Video Frame Enhancement Network
  - SFID: Single Frame Image Deblurring
- ADFN: A Multi-scale Flow-based Defect Detection Network
- Transformer: Temporal consistency modeling module

## Overall architecture

```text
Input video
  -> Single frame image sequence
  -> MEPEM
       ├─ VFE: video frame enhancement
       └─ SFID: single frame image deblurring
  -> Feature extraction
  -> ADFN: multi-scale flow-based spatial defect detection
  -> Raw frame-level anomaly scores
  -> Transformer temporal modeling
  -> Refined frame-level anomaly scores, metrics, and optional rendered video
```

This corresponds to the project architecture:

```text
                     Mine Environment Perception Enhancement Module
                 ┌────────────────────────────────────────────────────┐
Input video ──> Single frame image ──> VFE ──> SFID ───────────────┐  │
                 └────────────────────────────────────────────────┘  │
                                                                    ↓
                                                        Feature extraction
                                                                    ↓
                                    ADFN multi-scale flow-based defect detector
                                                                    ↓
                                                      Transformer temporal model
```

## Project structure

```text
MEPEM_ADFN_Transformer/
  configs/
    pipeline.example.yaml
  data/
    videos/                         # input videos, user-provided
    images/                         # optional ordered frame images
  docs/
    pipeline_interface.md
  modules/
    mepem/
      vfe/                          # Video Frame Enhancement Network
      sfid/                         # Single Frame Image Deblurring
    adfn/
      main.py                       # ADFN training/testing entry
      temporal/
        preprocessing.py            # MEPEM -> ADFN frame adapter
        export_adfn_scores.py       # video -> raw ADFN frame scores
        export_adfn_image_scores.py # image sequence -> raw ADFN frame scores
        compare_temporal_models.py  # Transformer/LSTM/GRU comparison
  outputs/
  scripts/
    run_video_pipeline.ps1
    run_video_pipeline.sh
  weights/
    vfe/
    sfid/
    adfn/
  requirements.txt
```

## Input and output

### Input

The project supports two input modes:

1. Continuous videos, such as `.mp4`, `.avi`, `.mov`, `.mkv`.
2. Ordered image frames with metadata.

The default processing image size is `512 × 512`. Original frames may have higher resolution; resizing/downsampling is performed before entering ADFN.

### Output

The standard pipeline produces:

```text
outputs/pipeline/
  raw_score_sequences.csv           # ADFN frame-wise anomaly scores
  model_comparison/
    transformer/
      checkpoint/best.pt
      training.log
      transformer_predictions.csv
    lstm/
    gru/
    model_comparison.csv
```

Typical reported metrics include:

- AUC-ROC
- FAR
- Fluctuation J
- FPS

## Runtime requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

The full inference pipeline requires:

```text
data/videos/
weights/vfe/vfe_frame_enhancement.pt       # optional, enables VFE
weights/sfid/sfid_deblurring.pt            # optional, enables SFID
weights/adfn/best_det.pt                   # required for ADFN scoring
```

If VFE or SFID weights are not provided, the corresponding MEPEM branch is skipped. ADFN weights are required for frame-level scoring unless you train ADFN first.

## Run the complete video pipeline

Linux or server:

```bash
bash scripts/run_video_pipeline.sh \
  --video-dir data/videos \
  --adfn-checkpoint weights/adfn/best_det.pt \
  --vfe-weights weights/vfe/vfe_frame_enhancement.pt \
  --sfid-checkpoint weights/sfid/sfid_deblurring.pt \
  --output-dir outputs/pipeline \
  --device cuda
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_video_pipeline.ps1 `
  -VideoDir .\data\videos `
  -AdfnCheckpoint .\weights\adfn\best_det.pt `
  -VfeWeights .\weights\vfe\vfe_frame_enhancement.pt `
  -SfidCheckpoint .\weights\sfid\sfid_deblurring.pt `
  -OutputDir .\outputs\pipeline `
  -Device cuda
```

Only run ADFN + Transformer without MEPEM:

```bash
bash scripts/run_video_pipeline.sh \
  --video-dir data/videos \
  --adfn-checkpoint weights/adfn/best_det.pt \
  --output-dir outputs/pipeline \
  --device cuda
```

## Train or test ADFN

Enter the ADFN module:

```bash
cd modules/adfn
```

Example training command:

```bash
python main.py \
  --dataset wirerope \
  --data-path ../../data \
  --input-size 512 512 \
  --batch-size 8 \
  --device cuda \
  --work-dir ../../outputs/adfn_training
```

Example testing command with an existing checkpoint:

```bash
python main.py \
  --dataset wirerope \
  --mode test \
  --eval_ckpt ../../weights/adfn/best_det.pt \
  --data-path ../../data \
  --input-size 512 512 \
  --device cuda \
  --work-dir ../../outputs/adfn_test
```

## Export ADFN frame scores

Video input:

```bash
cd modules/adfn

python temporal/export_adfn_scores.py \
  --video-dir ../../data/videos \
  --checkpoint ../../weights/adfn/best_det.pt \
  --output ../../outputs/pipeline/raw_score_sequences.csv \
  --device cuda \
  --input-size 512 512 \
  --vfe-root ../mepem/vfe \
  --vfe-weights ../../weights/vfe/vfe_frame_enhancement.pt \
  --sfid-root ../mepem/sfid \
  --sfid-checkpoint ../../weights/sfid/sfid_deblurring.pt
```

Image sequence input:

```bash
python temporal/export_adfn_image_scores.py \
  --image-root ../../data/images \
  --checkpoint ../../weights/adfn/best_det.pt \
  --output ../../outputs/pipeline/raw_score_sequences.csv \
  --device cuda \
  --input-size 512 512
```

## Train and compare temporal models

```bash
cd modules/adfn

python temporal/compare_temporal_models.py \
  --score-dir ../../outputs/pipeline/raw_score_sequences.csv \
  --output-dir ../../outputs/pipeline/model_comparison \
  --device cuda \
  --models transformer lstm gru
```

The Transformer module receives ADFN frame-wise anomaly scores rather than raw image frames. The expected temporal input is a score sequence window with shape equivalent to:

```text
[batch, T, 1]
```

Overlapping windows are restored to complete video sequences by averaging duplicated frame predictions.

