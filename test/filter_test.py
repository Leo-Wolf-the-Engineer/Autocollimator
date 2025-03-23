import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
from pathlib import Path

# Import project modules
from src.image_filter.image_filter import ImageFilter
from src.image_processing.image_processing import ImageProcessor
from src.image_aquisition import CameraManager

def test_image_filters(video_path, frame_number=0, filter_configs=None):
    """
    Test various image filters on a single frame and display results

    Args:
        video_path: Path to AVI video file
        frame_number: Frame number to process (default: 0)
        filter_configs: List of filter configurations to test
    """
    if filter_configs is None:
        # Default filter configurations if none provided
        filter_configs = [
            {"name": "Original", "type": None, "params": {}},
            {"name": "Bilateral", "type": "Bilateral", "params": {"d": 9, "sigma_color": 75, "sigma_space": 75}},
            {"name": "Background", "type": "Background", "params": {"method": "square_and_divide"}},
            {"name": "Gaussian", "type": "Gaussian", "params": {"sigma": 1, "kernel_size": 19}},
            {"name": "Median", "type": "Median", "params": {"kernel_size": 5}},
            {"name": "Morphological", "type": "Morphological", "params": {"operation": "erosion", "kernel_size": 5}},
            {"name": "Fourier", "type": "Fourier", "params": {"cutoff": 0.5}}
        ]
    # Initialize camera with specified video
    camera = CameraManager("AVI", video_path=video_path)
    width, height = camera.get_image_size()

    # Jump to specified frame
    for _ in range(frame_number):
        camera.retrieve_frame()

    # Get the frame
    original_frame = camera.retrieve_frame()
    camera.close()

    # Apply each filter
    filtered_frames = []
    for config in filter_configs:
        filter_type = config["type"]
        filter_params = config["params"]
        filter_name = config["name"]

        if filter_type is None and filter_name == "Original":
            # Original frame (no filter)
            filtered_frames.append({
                "name": "Original",
                "frame": original_frame,
                "params": {}
            })
        else:
            # Apply filter
            try:
                image_filter = ImageFilter(filter_type=filter_type, **filter_params)
                filtered = image_filter.apply(original_frame)
                filtered_frames.append({
                    "name": filter_name,
                    "frame": filtered,
                    "params": filter_params
                })
            except Exception as e:
                print(f"Error applying {filter_name} filter: {str(e)}")

    # Create subplot layout
    num_filters = len(filtered_frames)
    cols = 3  # 3 images per row
    rows = (num_filters + cols - 1) // cols  # Ceiling division

    # Create descriptive titles with parameters
    subplot_titles = []
    for f in filtered_frames:
        if f["name"] == "Original":
            subplot_titles.append("Original")
        else:
            # Format parameters as string
            params_str = ", ".join([f"{k}={v}" for k, v in f["params"].items()])
            subplot_titles.append(f"{f['name']}: {params_str}")

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.0,  # Reduce horizontal spacing between subplots
        vertical_spacing=0.0     # Reduce vertical spacing between subplots
    )

    # Add images to subplots
    for i, f in enumerate(filtered_frames):
        row = i // cols + 1
        col = i % cols + 1

        # Convert to RGB for Plotly if needed
        frame_rgb = f["frame"]
        if len(frame_rgb.shape) == 2:  # Grayscale
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2RGB)
        elif frame_rgb.shape[2] == 3:  # BGR
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)

        fig.add_trace(
            go.Image(z=frame_rgb),
            row=row, col=col
        )

    # Update layout
    fig.update_layout(
        title_text=f"Filter Comparison - Frame {frame_number}",
        height=800 * rows,  # Increase height per row
        width=3200,
        font=dict(size=18),  # Smaller font size for all text
    )

    # Make subplot titles smaller
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=10)  # Smaller font size for subplot titles

    # Remove axes for cleaner look
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)

    # Display the figure
    fig.show()

    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test image filters on video frame')
    parser.add_argument('--video', type=str, default='Videos/6.avi', help='Path to AVI video file')
    parser.add_argument('--frame', type=int, default=0, help='Frame number to process')
    args = parser.parse_args()

    # Example of custom filter configurations
    # You can modify this or pass custom configs to the function
    filter_configs = [
        {"name": "Original", "type": None, "params": {}},
        {"name": "Background", "type": "Background", "params": {"method": "mean"}},
        {"name": "Background", "type": "Background", "params": {"method": "median"}},
        {"name": "Background", "type": "Background", "params": {"method": "square_and_divide"}},
        {"name": "Bilateral", "type": "Bilateral", "params": {"d": 7, "sigma_color": 150, "sigma_space": 10}},
        {"name": "Bilateral", "type": "Bilateral", "params": {"d": 9, "sigma_color": 75, "sigma_space": 25}},
        {"name": "Bilateral", "type": "Bilateral", "params": {"d": 15, "sigma_color": 150, "sigma_space": 10}},
        {"name": "Fourier", "type": "Fourier", "params": {"type": "lowpass", "cutoff": 0.05}},
        #{"name": "Fourier", "type": "Fourier", "params": {"type": "highpass", "cutoff": 0.05}},
        {"name": "Gaussian", "type": "Gaussian", "params": {"sigma": 2, "kernel_size": 9}},
        {"name": "Median", "type": "Median", "params": {"kernel_size": 15}},
        {"name": "Morphological", "type": "Morphological", "params": {"operation": "closing", "kernel_size": 15}},
        {"name": "Morphological", "type": "Morphological", "params": {"operation": "dilation", "kernel_size": 15}},
        {"name": "Morphological", "type": "Morphological", "params": {"operation": "erosion", "kernel_size": 5}},
        {"name": "Morphological", "type": "Morphological", "params": {"operation": "erosion", "kernel_size": 11}},
        {"name": "Morphological", "type": "Morphological", "params": {"operation": "gradient", "kernel_size": 25}},
        {"name": "Morphological", "type": "Morphological", "params": {"operation": "opening", "kernel_size": 9}},

    ]

    test_image_filters(args.video, args.frame, filter_configs)