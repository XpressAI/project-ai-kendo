import os
from pose_analyzer import process_kendo_analysis_on_dir

if __name__ == "__main__":
    input_dir = "input_videos"   # Directory containing individual cut videos
    output_dir = "output_results"
    temp_dir = "temp"  # You can specify or leave None

    process_kendo_analysis_on_dir(input_dir, output_dir, temp_dir)
