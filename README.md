# Project AI Kendo

This is a fun side project that combines my love for Kendo and technology. Using tools like Mediapipe and `EVF-SAM2`, it analyzes Kendo movements to create cool visualizations and insights. "AI Kendo" pays homage to both the technology driving this project (AI) and the dojo, Ai Kendo, that inspired it.

## Features

- **Pose Estimation**: Uses Mediapipe to analyze Kendo stances and movements.
- **Segmentation**: Applies `EVF-SAM2` to generate segmentation masks.
- **Video Processing**: Generates multiple visualization outputs:
  - Original vs Pose estimation.
  - Original vs Segmentation.
  - Masked-only video.
  - Combined 4-quadrant video.
- **Analysis Results**: Outputs JSON files containing frame-by-frame and overall analysis.

## Installation

1. Clone the repository:

   ```bash
   https://github.com/XpressAI/project-ai-kendo
   cd project-ai-kendo
   ```

2. Run the setup script:

   ```bash
   ./setup.sh
   ```

   The script will:
   - Initialize the `evf-sam` submodule.
   - Install Python dependencies.
   - Download the required model checkpoints.

## Usage

### Process Videos

1. Place your Kendo videos in the `input_videos/` folder.
2. Run the main pipeline:

   ```bash
   python backend/main.py
   ```

3. Results will be saved in the `output_results/` folder.

### Run Segmentation as a Standalone Script

```bash
python backend/evf_sam2_inference.py \
    --version EVF-SAM/checkpoints/evf_sam2 \
    --input_folder ./input_frames \
    --output_folder ./output_results \
    --prompt "A shinai (竹刀) is a Japanese sword..." \
    --model_type sam2 \
    --precision fp16
```

### Output Directory

Each processed video will have its own folder under `output_results/`. The folder will contain:
- Extracted frames.
- Visualization videos.
- JSON files with analytical data.

## Outputs

### Generated Videos

1. **`processed.mp4`**: Combined side-by-side (original | pose).
2. **`pose_only.mp4`**: Video with only pose visualization.
3. **`masked_only.mp4`**: Video with original frames and mask overlay.
4. **`final_combined.mp4`**: 4-quadrant video:
   - Top-left: Original
   - Top-right: Pose
   - Bottom-left: Segmented
   - Bottom-right: Combined (original + pose + segmentation mask overlay)
5. **`original_pose.mp4`**: Side-by-side original and pose estimation.
6. **`original_segmented.mp4`**: Side-by-side original and segmentation visualization.

### Analytical Results

- **`analysis_results.json`**: Contains detailed frame-by-frame and overall analysis:
  - `initial_frame`: First frame analyzed.
  - `final_frame`: Last frame analyzed.
  - `initial_shinai_angle`: Shinai angle at the start.
  - `final_shinai_angle`: Shinai angle at the end.
  - `cut_classification`: Classification of the cut (`big_cut` or `small_cut`).

## Acknowledgements

- **The EVF-SAM2 Team**: Big thanks to the folks behind EVF-SAM2 for their amazing work on segmentation. Their model powers this project. Check it out [here](https://github.com/hustvl/EVF-SAM).


## License

This project is licensed under the [MIT License](LICENSE).
