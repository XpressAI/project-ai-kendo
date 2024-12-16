import os
from pose_analyzer import process_kendo_analysis_on_dir

if __name__ == "__main__":
    input_dir = "input_videos"   # Directory containing already cut videos of individual swings
    output_dir = "output_results"
    temp_dir = None  # If None, it will default inside the function

    # Run the analysis
    process_kendo_analysis_on_dir(input_dir, output_dir, temp_dir)
