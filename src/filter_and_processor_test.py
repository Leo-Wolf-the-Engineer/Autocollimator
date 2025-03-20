import numpy as np
import cv2
import logging
import time
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pandas as pd
#import sys
#from os.path import dirname, abspath

# Add the project root directory to the Python path
#project_root = dirname(dirname(abspath(__file__)))
#sys.path.append(project_root)

# Now import your project modules
from src.image_filter.image_filter import ImageFilter
from src.image_processing.image_processing import ImageProcessor
from src.image_aquisition import CameraManager

def test_filter_processor_stability():
    """Test all filter and processor combinations for stability using 500 frames"""
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s')

    # Create results directory
    results_dir = Path("../test/filter_test_results")
    results_dir.mkdir(exist_ok=True)

    # Initialize camera
    camera = CameraManager("AVI")
    width, height = camera.get_image_size()

    # Define filter configurations
    filter_configs = [
        {"type": None, "params": {}},  # No filter
        {"type": "Gaussian", "params": {"sigma": 1.5, "kernel_size": 5}},
        {"type": "Median", "params": {"kernel_size": 5}},
        {"type": "Bilateral", "params": {"d": 9, "sigma_color": 75, "sigma_space": 75}},
        {"type": "Background", "params": {"method": "mean"}},
        {"type": "Morphological", "params": {"operation": "opening", "kernel_size": 5}},
        {"type": "Fourier", "params": {}}
    ]

    # Define processor types
    processor_types = ["FastGaussian", "AccurateGaussian", "Peakfinder", "Linefit", "Dummy"]

    # Store results
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

            logging.info(f"Testing combination: {combo_name}")

            # Store positions
            x_positions = []
            y_positions = []

            # Process frames
            for i in range(num_frames):
                if i % 50 == 0:
                    logging.info(f"Processing frame {i} of {num_frames} for {combo_name}")

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
                        logging.warning(f"No position detected for frame {i} in {combo_name}")
                except Exception as e:
                    logging.error(f"Processing error on frame {i}: {str(e)}")

            # Calculate stability metrics if we have enough data
            if len(x_positions) > 10:
                x_positions = np.array(x_positions)
                y_positions = np.array(y_positions)

                # Calculate RMS from mean position (stability metric)
                x_mean = np.mean(x_positions)
                y_mean = np.mean(y_positions)

                x_rms = np.sqrt(np.mean((x_positions - x_mean)**2))
                y_rms = np.sqrt(np.mean((y_positions - y_mean)**2))
                combined_rms = np.sqrt(x_rms**2 + y_rms**2)

                # Store results
                all_results[combo_name] = {
                    "x_rms": x_rms,
                    "y_rms": y_rms,
                    "combined_rms": combined_rms,
                    "detection_rate": len(x_positions) / num_frames,
                    "x_positions": x_positions,
                    "y_positions": y_positions
                }

                logging.info(f"Results for {combo_name}: RMS_X={x_rms:.3f}, RMS_Y={y_rms:.3f}, "
                           f"Combined={combined_rms:.3f}, Detection={len(x_positions)/num_frames:.1%}")
            else:
                logging.warning(f"Insufficient data for {combo_name}")
                all_results[combo_name] = {
                    "x_rms": float('nan'),
                    "y_rms": float('nan'),
                    "combined_rms": float('nan'),
                    "detection_rate": len(x_positions) / num_frames
                }

    # Close camera
    camera.close()

    # Generate summary report
    generate_stability_report(all_results, results_dir)

    return all_results

def generate_stability_report(results: Dict, output_dir: Path):
    """Generate comprehensive report of stability tests"""
    # Create summary dataframe
    data = []
    for combo_name, metrics in results.items():
        data.append({
            "Combination": combo_name,
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

    # Generate plots for valid combinations
    for combo_name, metrics in results.items():
        if "x_positions" in metrics and len(metrics["x_positions"]) > 10:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

            # Plot X positions over time
            ax1.plot(metrics["x_positions"])
            ax1.set_title(f"{combo_name} - X Position")
            ax1.set_xlabel("Frame")
            ax1.set_ylabel("X Position (pixels)")

            # Plot Y positions over time
            ax2.plot(metrics["y_positions"])
            ax2.set_title(f"{combo_name} - Y Position")
            ax2.set_xlabel("Frame")
            ax2.set_ylabel("Y Position (pixels)")

            # Plot 2D scatter plot
            ax3.scatter(metrics["x_positions"], metrics["y_positions"], alpha=0.5)
            ax3.set_title(f"{combo_name} - Position Scatter")
            ax3.set_xlabel("X Position (pixels)")
            ax3.set_ylabel("Y Position (pixels)")
            ax3.axis('equal')

            # Add RMS info
            ax3.text(0.05, 0.95,
                    f"RMS X: {metrics['x_rms']:.3f}\nRMS Y: {metrics['y_rms']:.3f}\n"
                    f"Combined: {metrics['combined_rms']:.3f}\n"
                    f"Detection: {metrics['detection_rate']:.1%}",
                    transform=ax3.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.savefig(output_dir / f"{combo_name}_stability.png")
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