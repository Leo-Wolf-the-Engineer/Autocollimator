import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List
import pandas as pd
import datetime
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Import project modules
from src.image_filter.image_filter import ImageFilter
from src.image_processing.image_processing import ImageProcessor
from src.image_aquisition import CameraManager

def test_filter_processor_stability():
    """Test all filter and processor combinations for stability using multiple camera configurations"""
    # Create timestamp for folder name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"filter_test_results_{timestamp}")
    results_dir.mkdir(exist_ok=True)

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

    # Define filter configurations
    filter_configs = [
        {"type": None, "params": {}},  # No filter
        {"type": "Gaussian", "params": {"sigma": 1, "kernel_size": 5}},
        {"type": "Gaussian", "params": {"sigma": 1.5, "kernel_size": 5}},
        {"type": "Gaussian", "params": {"sigma": 2, "kernel_size": 5}},
        {"type": "Gaussian", "params": {"sigma": 2.5, "kernel_size": 5}},
        {"type": "Gaussian", "params": {"sigma": 3, "kernel_size": 5}},
        #{"type": "Median", "params": {"kernel_size": 5}},
        #{"type": "Bilateral", "params": {"d": 9, "sigma_color": 75, "sigma_space": 75}},
        #{"type": "Background", "params": {"method": "mean"}},
        #{"type": "Morphological", "params": {"operation": "opening", "kernel_size": 5}},
        #{"type": "Fourier", "params": {}}
    ]

    # Define processor types
    processor_types = ["FastGaussian", "AccurateGaussian", "Peakfinder", "Linefit"] #, "Dummy"

    # Save test settings
    save_test_settings(results_dir, filter_configs, processor_types)

    # Store all results across camera settings
    combined_results = {}

    # Test with camera settings 1-9
    for camera_setting in range(1, 2):
        logging.info(f"Testing with file {camera_setting}")

        # Initialize camera with specified setting
        camera = CameraManager("AVI", video_path=f'Videos/{camera_setting}.avi')
        width, height = camera.get_image_size()

        # Store results for this camera setting
        all_results = {}

        # Number of frames to test
        num_frames = 50

        # Process all combinations
        for proc_type in processor_types:
            processor = ImageProcessor(proc_type, width, height)

            for filter_config in filter_configs:
                filter_type = filter_config["type"]
                filter_params = filter_config["params"]

                # Create filter (or None)
                if filter_type:
                    image_filter = ImageFilter(filter_type, **filter_params)
                    combo_name = f"{filter_type}_{proc_type}"
                else:
                    image_filter = None
                    combo_name = f"NoFilter_{proc_type}"

                combo_setting_name = f"{combo_name}_setting{camera_setting}"
                logging.info(f"Testing combination: {combo_setting_name}")

                # Store positions
                x_positions = []
                y_positions = []

                # Process frames
                for i in range(num_frames):
                    if i % 10 == 0:
                        logging.info(f"Processing frame {i} of {num_frames} for {combo_setting_name}")

                    # Get frame from camera
                    frame = camera.retrieve_frame()

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

                        # Store the first detected point if available
                        if len(x_vals) > 0 and len(y_vals) > 0:
                            x_positions.append(x_vals[0])
                            y_positions.append(y_vals[0])
                        else:
                            logging.warning(f"No position detected for frame {i} in {combo_setting_name}")
                    except Exception as e:
                        logging.error(f"Processing error on frame {i}: {str(e)}")

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
                        "filter_type": filter_type if filter_type else "NoFilter",
                        "processor_type": proc_type,
                        "camera_setting": camera_setting,
                        "x_rms": x_rms,
                        "y_rms": y_rms,
                        "combined_rms": combined_rms,
                        "detection_rate": len(x_positions) / num_frames,
                        "x_positions": x_positions,
                        "y_positions": y_positions
                    }

                    logging.info(f"Results for {combo_setting_name}: RMS_X={x_rms:.3f}, RMS_Y={y_rms:.3f}, "
                                f"Combined={combined_rms:.3f}, Detection={len(x_positions)/num_frames:.1%}")
                else:
                    logging.warning(f"Insufficient data for {combo_setting_name}")
                    all_results[combo_setting_name] = {
                        "filter_type": filter_type if filter_type else "NoFilter",
                        "processor_type": proc_type,
                        "camera_setting": camera_setting,
                        "x_rms": float('nan'),
                        "y_rms": float('nan'),
                        "combined_rms": float('nan'),
                        "detection_rate": len(x_positions) / num_frames
                    }

        # Add this camera setting's results to the combined results
        combined_results.update(all_results)

        # Close camera
        camera.close()

    # Generate summary report
    generate_stability_report(combined_results, results_dir)

    return combined_results

def save_test_settings(output_dir: Path, filter_configs: List, processor_types: List):
    """Save test settings to a text file"""
    with open(output_dir / "test_settings.txt", "w") as f:
        f.write("Filter and Processor Stability Test Settings\n")
        f.write("==========================================\n\n")

        f.write("Test Date and Time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

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
    """Generate comprehensive report of stability tests with combined plots"""
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
            "Detection Rate": metrics["detection_rate"]
        })

    df = pd.DataFrame(data)

    # Sort by combined RMS (lower is better)
    df_sorted = df.sort_values("Combined RMS")

    # Save to CSV
    df_sorted.to_csv(output_dir / "stability_results.csv", index=False)

    # Create single plot with all position data
    plt.figure(figsize=(15, 10))

    # Get unique filter and processor combinations
    unique_combinations = set()
    for combo in results.keys():
        base_combo = '_'.join(combo.split('_')[:-1])  # Remove setting part
        unique_combinations.add(base_combo)

    # Plot all X positions
    plt.subplot(2, 1, 1)
    for base_combo in unique_combinations:
        for setting in range(1, 10):
            combo_name = f"{base_combo}_setting{setting}"
            if combo_name in results and "x_positions" in results[combo_name]:
                plt.plot(results[combo_name]["x_positions"],
                        alpha=0.7,
                        label=f"{combo_name}")

    plt.title("X Positions for All Combinations")
    plt.xlabel("Frame")
    plt.ylabel("X Position (pixels)")
    plt.grid(True, alpha=0.3)

    # Plot all Y positions
    plt.subplot(2, 1, 2)
    for base_combo in unique_combinations:
        for setting in range(1, 10):
            combo_name = f"{base_combo}_setting{setting}"
            if combo_name in results and "y_positions" in results[combo_name]:
                plt.plot(results[combo_name]["y_positions"],
                        alpha=0.7,
                        label=f"{combo_name}")

    plt.title("Y Positions for All Combinations")
    plt.xlabel("Frame")
    plt.ylabel("Y Position (pixels)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "all_positions.png")
    plt.close()

    # Create a 3D plot for RMS values
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Extract unique filter types and processor types
    filter_types = sorted(list(df["Filter"].unique()))
    processor_types = sorted(list(df["Processor"].unique()))

    # Create coordinate matrices
    x_indices = np.arange(len(filter_types))
    y_indices = np.arange(len(processor_types))
    x_mesh, y_mesh = np.meshgrid(x_indices, y_indices)

    # Create RMS value matrix
    z_values = np.zeros((len(processor_types), len(filter_types)))
    z_values.fill(np.nan)  # Fill with NaN to handle missing combinations

    # Calculate average RMS for each filter-processor combo across camera settings
    for i, processor in enumerate(processor_types):
        for j, filter_type in enumerate(filter_types):
            # Get data for this combination across all camera settings
            combo_data = df[(df["Filter"] == filter_type) & (df["Processor"] == processor)]
            if not combo_data.empty:
                # Average the RMS values across camera settings
                z_values[i, j] = combo_data["Combined RMS"].mean()

    # Plot the 3D surface
    surf = ax.plot_surface(x_mesh, y_mesh, z_values, cmap=cm.coolwarm,
                         linewidth=0, antialiased=True, alpha=0.8)

    # Set labels
    ax.set_xlabel('Filter Type')
    ax.set_ylabel('Processor Type')
    ax.set_zlabel('Average RMS')
    ax.set_title('3D Visualization of Stability (RMS) by Filter and Processor')

    # Set axis ticks
    ax.set_xticks(x_indices)
    ax.set_xticklabels(filter_types, rotation=45)
    ax.set_yticks(y_indices)
    ax.set_yticklabels(processor_types)

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.savefig(output_dir / "3d_rms_visualization.png")
    plt.close()

    # Create summary text report
    with open(output_dir / "stability_summary.txt", "w") as f:
        f.write("Filter and Processor Stability Test Results\n")
        f.write("==========================================\n\n")

        f.write("Combinations sorted by stability (lowest RMS first):\n")
        for _, row in df_sorted.iterrows():
            if not np.isnan(row["Combined RMS"]):
                f.write(f"{row['Combination']}: RMS={row['Combined RMS']:.3f}, "
                      f"Detection Rate={row['Detection Rate']:.1%}\n")

        f.write("\n\nDetailed Results:\n")
        for combo_name, metrics in results.items():
            f.write(f"\n{combo_name}:\n")
            f.write(f"  RMS X: {metrics['x_rms']:.3f}\n")
            f.write(f"  RMS Y: {metrics['y_rms']:.3f}\n")
            f.write(f"  RMS Combined: {metrics['combined_rms']:.3f}\n")
            f.write(f"  Detection Rate: {metrics['detection_rate']:.1%}\n")

if __name__ == "__main__":
    test_filter_processor_stability()