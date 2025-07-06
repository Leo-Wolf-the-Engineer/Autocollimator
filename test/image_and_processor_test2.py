import unittest
import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple
from pathlib import Path

# Import filter classes
from src.image_filter.image_filter import ImageFilter

# Import processor classes
from src.image_processing.image_processing import ImageProcessor

class FilterProcessorTest(unittest.TestCase):
    """Test stability of all filter and processor combinations"""

    def setUp(self):
        """Set up test parameters"""
        self.width = 640
        self.height = 480
        self.num_frames = 50
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

        # Define filter types and their parameters
        self.filter_configs = [
            {"type": None, "params": {}},  # No filter
            {"type": "Gaussian", "params": {"sigma": 1.5, "kernel_size": 5}},
            {"type": "Median", "params": {"kernel_size": 5}},
            {"type": "Bilateral", "params": {"d": 9, "sigma_color": 75, "sigma_space": 75}},
            {"type": "Background", "params": {"method": "mean"}},
            {"type": "Morphological", "params": {"operation": "opening", "kernel_size": 5}},
            {"type": "Fourier", "params": {}}
        ]

        # Define processor types
        self.processor_types = ["FastGaussian", "AccurateGaussian", "Peakfinder", "Linefit", "Dummy"]

    def generate_test_frame(self, frame_num: int) -> np.ndarray:
        """Generate a test frame with a Gaussian spot at a known position"""
        # Create blank frame
        frame = np.zeros((self.height, self.width), dtype=np.uint8)

        # Known position with slight sinusoidal movement
        true_x = self.width // 2 + int(10 * np.sin(frame_num / 50))
        true_y = self.height // 2 + int(5 * np.cos(frame_num / 30))

        # Add Gaussian spot
        spot_size = 50
        y, x = np.ogrid[-spot_size:spot_size+1, -spot_size:spot_size+1]
        spot = np.exp(-(x*x + y*y) / (2 * 5**2)) * 255

        # Place spot at position
        y_min = max(0, true_y - spot_size)
        y_max = min(self.height, true_y + spot_size + 1)
        x_min = max(0, true_x - spot_size)
        x_max = min(self.width, true_x + spot_size + 1)

        spot_y_min = max(0, spot_size - true_y)
        spot_y_max = spot_size*2 + 1 - max(0, (true_y + spot_size + 1) - self.height)
        spot_x_min = max(0, spot_size - true_x)
        spot_x_max = spot_size*2 + 1 - max(0, (true_x + spot_size + 1) - self.width)

        # Add spot to frame
        frame[y_min:y_max, x_min:x_max] = spot[spot_y_min:spot_y_max, spot_x_min:spot_x_max]

        # Add some random noise
        noise = np.random.normal(0, 5, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return frame, (true_x - self.width/2, true_y - self.height/2)  # Return frame and true position

    def calculate_rms(self, measured: np.ndarray, expected: np.ndarray) -> float:
        """Calculate RMS error between measured and expected values"""
        return np.sqrt(np.mean((measured - expected) ** 2))

    def test_all_combinations(self):
        """Test all filter and processor combinations"""
        results = {}

        # Loop through all processor types
        for proc_type in self.processor_types:
            processor = ImageProcessor(proc_type, self.width, self.height)

            # Loop through all filter types
            for filter_config in self.filter_configs:
                filter_type = filter_config["type"]
                filter_params = filter_config["params"]

                # Create filter (or None for no filter)
                if filter_type:
                    image_filter = ImageFilter(filter_type, **filter_params)
                    combo_name = f"{filter_type}_{proc_type}"
                else:
                    image_filter = None
                    combo_name = f"NoFilter_{proc_type}"

                logging.info(f"Testing combination: {combo_name}")

                # Store positions
                positions_x = []
                positions_y = []
                true_positions_x = []
                true_positions_y = []

                # Process frames
                for i in range(self.num_frames):
                    # Generate frame with known position
                    frame, (true_x, true_y) = self.generate_test_frame(i)

                    # Apply filter if available
                    if image_filter:
                        filtered_frame = image_filter.apply(frame)
                    else:
                        filtered_frame = frame

                    # Process frame
                    try:
                        x_values, y_values = processor.process_frame(filtered_frame)

                        # Take the first detected point if multiple are returned
                        if len(x_values) > 0 and len(y_values) > 0:
                            positions_x.append(x_values[0])
                            positions_y.append(y_values[0])
                            true_positions_x.append(true_x)
                            true_positions_y.append(true_y)
                        else:
                            logging.warning(f"No valid position detected for frame {i} in {combo_name}")
                    except Exception as e:
                        logging.error(f"Error processing frame {i} with {combo_name}: {str(e)}")

                # Convert to numpy arrays
                positions_x = np.array(positions_x)
                positions_y = np.array(positions_y)
                true_positions_x = np.array(true_positions_x)
                true_positions_y = np.array(true_positions_y)

                # Calculate RMS if we have data
                if len(positions_x) > 0:
                    rms_x = self.calculate_rms(positions_x, true_positions_x)
                    rms_y = self.calculate_rms(positions_y, true_positions_y)
                    rms_combined = np.sqrt(rms_x**2 + rms_y**2)

                    # Store results
                    results[combo_name] = {
                        "rms_x": rms_x,
                        "rms_y": rms_y,
                        "rms_combined": rms_combined,
                        "detected_frames": len(positions_x),
                        "total_frames": self.num_frames
                    }

                    logging.info(f"Results for {combo_name}: RMS_X={rms_x:.2f}, RMS_Y={rms_y:.2f}, "
                                f"Combined={rms_combined:.2f}, Detection Rate={len(positions_x)/self.num_frames:.1%}")
                else:
                    logging.warning(f"No valid positions detected for {combo_name}")
                    results[combo_name] = {
                        "rms_x": float('nan'),
                        "rms_y": float('nan'),
                        "rms_combined": float('nan'),
                        "detected_frames": 0,
                        "total_frames": self.num_frames
                    }

        # Output summary of results
        self.output_results_summary(results)

    def output_results_summary(self, results: Dict):
        """Output summary of test results"""
        # Sort combinations by RMS
        sorted_results = sorted(
            [(k, v) for k, v in results.items() if not np.isnan(v["rms_combined"])],
            key=lambda x: x[1]["rms_combined"]
        )

        # Write to file
        with open(self.results_dir / "filter_processor_results.txt", "w") as f:
            f.write("Filter and Processor Stability Test Results\n")
            f.write("==========================================\n\n")

            f.write("Combinations sorted by stability (lowest RMS first):\n")
            for name, result in sorted_results:
                f.write(f"{name}: RMS={result['rms_combined']:.2f}, "
                       f"Detection Rate={result['detected_frames']/result['total_frames']:.1%}\n")

            f.write("\n\nDetailed Results:\n")
            for name, result in results.items():
                f.write(f"\n{name}:\n")
                f.write(f"  RMS X: {result['rms_x']:.2f}\n")
                f.write(f"  RMS Y: {result['rms_y']:.2f}\n")
                f.write(f"  RMS Combined: {result['rms_combined']:.2f}\n")
                f.write(f"  Detection Rate: {result['detected_frames']}/{result['total_frames']} "
                       f"({result['detected_frames']/result['total_frames']:.1%})\n")

if __name__ == "__main__":
    unittest.main()