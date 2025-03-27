import numpy as np
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
import datetime
import time
import cv2

# Import project modules
from src.image_filter.image_filter import ImageFilter
from src.image_processing.image_processing import ImageProcessor
from src.image_aquisition import CameraManager

def test_filter_processor_stability():
    """Test all filter and processor combinations for stability using multiple camera configurations
    Args:
        pixels_to_arcsec: Conversion factor from pixels to arcseconds (calibration factor)
    """
    PIXEL_PITCH = 3.45e-6  # in meters
    FOCAL_LENGTH = 0.385  # in meters
    pixels_to_arcsec = PIXEL_PITCH / (2 * FOCAL_LENGTH) * 180 / np.pi * 3600
    print(pixels_to_arcsec)
    # Create timestamp for folder name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"filter_test_results_{timestamp}")
    results_dir.mkdir(exist_ok=True)

    # Create a subdirectory for sample images
    image_dir = results_dir / "sample_images"
    image_dir.mkdir(exist_ok=True)

    # Setup logging
    log_file = results_dir / "test_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Log conversion factor
    logging.info(f"Using conversion factor: {pixels_to_arcsec} arcseconds/pixel")

    # Define filter configurations
    filter_configs = [
        {"name": "Original", "type": None, "params": {}},
        {"name": "Background_mean", "type": "Background", "params": {"method": "mean"}},
        {"name": "Background_median", "type": "Background", "params": {"method": "median"}},
        {"name": "Background_sd", "type": "Background", "params": {"method": "square_and_divide"}},
        {"name": "Bilateral_d7_c150_s10", "type": "Bilateral", "params": {"d": 7, "sigma_color": 150, "sigma_space": 10}},
        {"name": "Bilateral_d9_75_25", "type": "Bilateral", "params": {"d": 9, "sigma_color": 75, "sigma_space": 25}},
        {"name": "Bilateral_d15_c150_s10", "type": "Bilateral", "params": {"d": 15, "sigma_color": 150, "sigma_space": 10}},
        {"name": "Fourier_lp", "type": "Fourier", "params": {"type": "lowpass", "cutoff": 0.05}},
        {"name": "Fourier_hp", "type": "Fourier", "params": {"type": "highpass", "cutoff": 0.05}},
        {"name": "Gaussian", "type": "Gaussian", "params": {"sigma": 2, "kernel_size": 9}},
        {"name": "Median", "type": "Median", "params": {"kernel_size": 15}},
        {"name": "Morphological_cl", "type": "Morphological", "params": {"operation": "closing", "kernel_size": 15}},
        {"name": "Morphological_di", "type": "Morphological", "params": {"operation": "dilation", "kernel_size": 15}},
        {"name": "Morphological_er5", "type": "Morphological", "params": {"operation": "erosion", "kernel_size": 5}},
        {"name": "Morphological_er11", "type": "Morphological", "params": {"operation": "erosion", "kernel_size": 11}},
        {"name": "Morphological_er25", "type": "Morphological", "params": {"operation": "gradient", "kernel_size": 25}},
        {"name": "Morphological_op", "type": "Morphological", "params": {"operation": "opening", "kernel_size": 9}},
    ]

    filter_configs = [
        {"name": "Original", "type": None, "params": {}},
        {"name": "Morphological_er5", "type": "Morphological", "params": {"operation": "erosion", "kernel_size": 5}},
        {"name": "Morphological_er11", "type": "Morphological", "params": {"operation": "erosion", "kernel_size": 11}},
        {"name": "Morphological_er25", "type": "Morphological", "params": {"operation": "gradient", "kernel_size": 25}},
    ]

    # Define processor types
    processor_types = ["WeightedPeakfinder", "Peakfinder"] #,AccurateGaussian "Dummy" "FastGaussian", , "Peakfinder", "Linefit"

    # Save test settings
    save_test_settings(results_dir, filter_configs, processor_types, pixels_to_arcsec)

    # Store all results across camera settings
    combined_results = {}

    # Test with camera settings 1-9
    for camera_setting in [6,]:
        logging.info(f"Testing with file {camera_setting}")

        # Initialize camera with specified setting
        camera = CameraManager("AVI", video_path=f'Videos/{camera_setting}.avi')
        width, height = camera.get_image_size()

        # Store results for this camera setting
        all_results = {}

        # Number of frames to test
        num_frames = 250

        # Process all combinations
        for proc_type in processor_types:
            processor = ImageProcessor(proc_type, width, height)

            for filter_config in filter_configs:
                filter_type = filter_config["type"]
                filter_params = filter_config["params"]
                filter_name = filter_config["name"]

                # Create filter (or None)
                if filter_type:
                    # Fix: Use filter_type as keyword argument to avoid conflicts
                    image_filter = ImageFilter(filter_type=filter_type, **filter_params)
                    combo_name = f"{filter_name}_{proc_type}"
                else:
                    image_filter = None
                    combo_name = f"NoFilter_{proc_type}"

                combo_setting_name = f"{camera_setting}_{combo_name}"
                logging.info(f"Testing combination: {combo_setting_name}")

                # Store positions and timing data
                x_positions = []
                y_positions = []
                frame_times = []
                first_image_saved = False

                # Process frames
                for i in range(num_frames):
                    if i % 10 == 0:
                        logging.info(f"Processing frame {i} of {num_frames} for {combo_setting_name}")

                    # Get frame from camera
                    frame = camera.retrieve_frame()

                    # Start timing
                    start_time = time.time()

                    # Apply filter if available
                    if image_filter:
                        try:
                            filtered_frame = image_filter.apply(frame)
                        except Exception as e:
                            logging.error(f"Filter error on frame {i}: {str(e)}")
                            filtered_frame = frame
                    else:
                        filtered_frame = frame

                    # Process frame
                    try:
                        x_vals, y_vals = processor.process_frame(filtered_frame)

                        # End timing after processing is complete
                        end_time = time.time()
                        frame_times.append(end_time - start_time)

                        # Store the first detected point if available, convert to arcseconds
                        if len(x_vals) > 0 and len(y_vals) > 0:
                            x_positions.append(x_vals[0] * pixels_to_arcsec)
                            y_positions.append(y_vals[0] * pixels_to_arcsec)

                            # For the first frame where a position is detected, save a copy with the detected position marked
                            if not first_image_saved:
                                first_image_saved = True
                                marked_frame = filtered_frame.copy()
                                # Draw a small red dot at the detected position
                                cv2.circle(marked_frame, (int(x_vals[0]+width/2), int(y_vals[0]+height/2)), 5, (0, 0, 255), -1)
                                cv2.imwrite(str(image_dir / f"{combo_setting_name}_marked.png"), marked_frame)
                        else:
                            logging.warning(f"No position detected for frame {i} in {combo_setting_name}")
                    except Exception as e:
                        logging.error(f"Processing error on frame {i}: {str(e)}")
                        # End timing even if there's an error
                        end_time = time.time()
                        frame_times.append(end_time - start_time)

                # Calculate average processing time
                avg_processing_time = np.mean(frame_times) if frame_times else float('nan')
                fps = 1.0 / avg_processing_time if avg_processing_time > 0 else float('nan')

                logging.info(f"Average processing time for {combo_setting_name}: {avg_processing_time*1000:.2f} ms ({fps:.1f} FPS)")

                # Calculate stability metrics if we have enough data
                if len(x_positions) > 10:
                    x_positions = np.array(x_positions)
                    y_positions = np.array(y_positions)

                    # Calculate RMS from mean position
                    x_mean = np.mean(x_positions)
                    y_mean = np.mean(y_positions)

                    x_rms = np.sqrt(np.mean((x_positions - x_mean)**2))
                    y_rms = np.sqrt(np.mean((y_positions - y_mean)**2))
                    combined_rms = np.sqrt(x_rms**2 + y_rms**2)

                    # Store results
                    all_results[combo_setting_name] = {
                        "filter_type": filter_name if filter_type else "NoFilter",
                        "processor_type": proc_type,
                        "camera_setting": camera_setting,
                        "x_rms": x_rms,
                        "y_rms": y_rms,
                        "combined_rms": combined_rms,
                        "detection_rate": len(x_positions) / num_frames,
                        "x_positions": x_positions,
                        "y_positions": y_positions,
                        "avg_processing_time": avg_processing_time,
                        "fps": fps
                    }

                    logging.info(f"Results for {combo_setting_name}: RMS_X={x_rms:.3f} arcsec, RMS_Y={y_rms:.3f} arcsec, "
                                f"Combined={combined_rms:.3f} arcsec, Detection={len(x_positions)/num_frames:.1%}")
                else:
                    logging.warning(f"Insufficient data for {combo_setting_name}")
                    all_results[combo_setting_name] = {
                        "filter_type": filter_name if filter_type else "NoFilter",
                        "processor_type": proc_type,
                        "camera_setting": camera_setting,
                        "x_rms": float('nan'),
                        "y_rms": float('nan'),
                        "combined_rms": float('nan'),
                        "detection_rate": len(x_positions) / num_frames,
                        "avg_processing_time": avg_processing_time,
                        "fps": fps
                    }

        # Add this camera setting's results to the combined results
        combined_results.update(all_results)

        # Close camera
        camera.close()

    # Generate summary report
    generate_stability_report(combined_results, results_dir)

    return combined_results

def save_test_settings(output_dir: Path, filter_configs: List, processor_types: List, pixels_to_arcsec: float):
    """Save test settings to a text file"""
    with open(output_dir / "test_settings.txt", "w") as f:
        f.write("Filter and Processor Stability Test Settings\n")
        f.write("==========================================\n\n")

        f.write("Test Date and Time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

        # Add conversion factor information
        f.write(f"Pixel to Arcsecond Conversion: {pixels_to_arcsec} arcseconds/pixel\n\n")

        f.write("Filter Configurations:\n")
        for i, config in enumerate(filter_configs):
            f.write(f"{i+1}. Type: {config['type'] if config['type'] else 'None'}\n")
            f.write(f"   Parameters: {config['params']}\n")

        f.write("\nProcessor Types:\n")
        for i, proc_type in enumerate(processor_types):
            f.write(f"{i+1}. {proc_type}\n")

        f.write("\nCamera Settings: Testing settings 1 through 9\n")
        f.write("Frames per test: 50\n")

def generate_stability_report(results: Dict, output_dir: Path):
    """Generate comprehensive report of stability tests with interactive Plotly visualizations"""
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    # Create summary dataframe
    data = []
    for combo_name, metrics in results.items():
        data.append({
            "Combination": combo_name,
            "Filter": metrics["filter_type"],
            "Processor": metrics["processor_type"],
            "Camera Setting": metrics["camera_setting"],
            "X RMS": metrics["x_rms"],
            "Y RMS": metrics["y_rms"],
            "Combined RMS": metrics["combined_rms"],
            "Detection Rate": metrics["detection_rate"],
            "Avg Processing Time": metrics.get("avg_processing_time", float('nan')),
            "FPS": metrics.get("fps", float('nan'))
        })

    df = pd.DataFrame(data)

    # Sort by combined RMS (lower is better)
    df_sorted = df.sort_values("Combined RMS")

    # Save to CSV
    df_sorted.to_csv(output_dir / "stability_results.csv", index=False)

    # Create interactive plot with all position data
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("X Positions for All Combinations",
                                        "Y Positions for All Combinations",))

    # Get unique filter and processor combinations
    unique_combinations = set()
    for combo in results.keys():
        base_combo = '_'.join(combo.split('_')[:-1])  # Remove setting part
        unique_combinations.add(base_combo)

    # Add X position traces
    for combo_name, metrics in results.items():
        if "x_positions" in metrics:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(metrics["x_positions"]))),
                    y=metrics["x_positions"],
                    mode='lines',
                    name=combo_name,
                    legendgroup=combo_name,
                    opacity=0.7
                ),
                row=1, col=1
            )

    # Add Y position traces
    for combo_name, metrics in results.items():
        if "y_positions" in metrics:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(metrics["y_positions"]))),
                    y=metrics["y_positions"],
                    mode='lines',
                    name=combo_name,
                    legendgroup=combo_name,
                    showlegend=False,  # Don't duplicate in legend
                    opacity=0.7
                ),
                row=2, col=1
            )

    # Update layout
    fig.update_layout(
        height=1200,
        width=1600,
        title_text="Position Tracking for All Filter-Processor Combinations",
        showlegend=True,
        legend=dict(
            groupclick="toggleitem",
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    # Update axes - now using arcseconds units
    fig.update_xaxes(title_text="Frame", row=1, col=1)
    fig.update_yaxes(title_text="X Position (arcsec)", row=1, col=1)
    fig.update_xaxes(title_text="Frame", row=2, col=1)
    fig.update_yaxes(title_text="Y Position (arcsec)", row=2, col=1)

    # Save as HTML
    fig.write_html(output_dir / "all_positions.html")

    # Create heatmap of average RMS for filter and processor combinations
    filter_types = sorted(list(df["Filter"].unique()))
    processor_types = sorted(list(df["Processor"].unique()))

    # Create RMS value matrix
    z_values = np.zeros((len(processor_types), len(filter_types)))
    z_values.fill(np.nan)  # Fill with NaN to handle missing combinations

    # Calculate average RMS for each filter-processor combo across camera settings
    for i, processor in enumerate(processor_types):
        for j, filter_type in enumerate(filter_types):
            combo_data = df[(df["Filter"] == filter_type) & (df["Processor"] == processor)]
            if not combo_data.empty:
                z_values[i, j] = combo_data["Combined RMS"].mean()

    # Calculate FPS value matrix
    fps_values = np.zeros((len(processor_types), len(filter_types)))
    fps_values.fill(np.nan)  # Fill with NaN to handle missing combinations

    # Calculate average FPS for each filter-processor combo across camera settings
    for i, processor in enumerate(processor_types):
        for j, filter_type in enumerate(filter_types):
            combo_data = df[(df["Filter"] == filter_type) & (df["Processor"] == processor)]
            if not combo_data.empty and not combo_data["FPS"].isna().all():
                fps_values[i, j] = combo_data["FPS"].mean()

    # Calculate RMS/FPS ratio matrix (efficiency metric - lower is better)
    ratio_values = np.zeros((len(processor_types), len(filter_types)))
    ratio_values.fill(np.nan)  # Fill with NaN to handle missing combinations

    # Calculate ratio for each filter-processor combo across camera settings
    for i, processor in enumerate(processor_types):
        for j, filter_type in enumerate(filter_types):
            combo_data = df[(df["Filter"] == filter_type) & (df["Processor"] == processor)]
            if not combo_data.empty and not combo_data["FPS"].isna().all() and not combo_data["Combined RMS"].isna().all():
                # Calculate average ratio - lower is better (low RMS, high FPS)
                ratios = combo_data["Combined RMS"] / combo_data["FPS"]
                ratio_values[i, j] = ratios.mean()

    # Create figure with 3 subplots (one per row)
    fig_3d = make_subplots(rows=3, cols=1,
                          specs=[[{'type': 'surface'}], 
                                 [{'type': 'surface'}], 
                                 [{'type': 'surface'}]],
                          subplot_titles=('Efficiency (RMS/FPS) lower is better', 'Stability (RMS)', 'Performance (FPS)'))

    # Add RMS/FPS ratio surface plot (first row)
    fig_3d.add_trace(
        go.Surface(z=ratio_values, x=filter_types, y=processor_types, colorscale='Turbo',
                   showscale=True, 
                   colorbar=dict(x=1.0, y=0.85, len=0.2, thickness=15, title="Ratio"),
                   name="RMS/FPS Ratio"),
        row=1, col=1
    )
    
    # Add RMS surface plot (second row)
    fig_3d.add_trace(
        go.Surface(z=z_values, x=filter_types, y=processor_types, colorscale='Viridis',
                   showscale=True, 
                   colorbar=dict(x=1.0, y=0.5, len=0.2, thickness=15, title="arcsec"),
                   name="RMS (arcsec)"),
        row=2, col=1
    )

    # Add FPS surface plot (third row)
    fig_3d.add_trace(
        go.Surface(z=fps_values, x=filter_types, y=processor_types, colorscale='Plasma',
                   showscale=True, 
                   colorbar=dict(x=1.0, y=0.15, len=0.2, thickness=15, title="FPS"),
                   name="FPS"),
        row=3, col=1
    )

    # Update layout
    fig_3d.update_layout(
        title=f'3D Visualization of Efficiency, Stability and Performance, Sample Size={len(df)}',
        autosize=False,
        width=1200,  
        height=2000,  # Increased height for vertical arrangement
        margin=dict(l=65, r=100, b=65, t=90)  # Increased right margin for colorbars
    )

    # Update scene properties for the efficiency plot
    fig_3d.update_scenes(
        xaxis_title='Filter Type',
        yaxis_title='Processor Type',
        zaxis_title='RMS/FPS Ratio (lower is better)',
        row=1, col=1
    )

    # Update scene properties for RMS plot
    fig_3d.update_scenes(
        xaxis_title='Filter Type',
        yaxis_title='Processor Type',
        zaxis_title='Average RMS (arcsec)',
        row=2, col=1
    )

    fig_3d.update_scenes(
        xaxis_title='Filter Type',
        yaxis_title='Processor Type',
        zaxis_title='FPS',
        row=3, col=1
    )

    # Save 3D plot as HTML
    fig_3d.write_html(output_dir / "3d_rms_visualization.html")

    # Create summary text report
    with open(output_dir / "stability_summary.txt", "w") as f:
        f.write("Filter and Processor Stability Test Results\n")
        f.write("==========================================\n\n")

        f.write("Combinations sorted by stability (lowest RMS first):\n")
        for _, row in df_sorted.iterrows():
            if not np.isnan(row["Combined RMS"]):
                f.write(f"{row['Combination']}: RMS={row['Combined RMS']:.3f} arcsec, "
                      f"Detection Rate={row['Detection Rate']:.1%}\n")

        f.write("\n\nDetailed Results:\n")
        for combo_name, metrics in results.items():
            f.write(f"\n{combo_name}:\n")
            f.write(f"  RMS X: {metrics['x_rms']:.3f} arcsec\n")
            f.write(f"  RMS Y: {metrics['y_rms']:.3f} arcsec\n")
            f.write(f"  RMS Combined: {metrics['combined_rms']:.3f} arcsec\n")
            f.write(f"  Detection Rate: {metrics['detection_rate']:.1%}\n")
            if "avg_processing_time" in metrics:
                f.write(f"  Avg Processing Time: {metrics['avg_processing_time']*1000:.2f} ms\n")
            if "fps" in metrics:
                f.write(f"  FPS: {metrics['fps']:.1f}\n")

if __name__ == "__main__":
    test_filter_processor_stability()
