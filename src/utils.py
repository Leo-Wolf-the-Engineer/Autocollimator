import matplotlib.pyplot as plt
import cv2

# Function to plot the current frame and intensity distributions
def plot_current_frame(axs, frame, intensity_x, intensity_y, peak_x, peak_y, peak_x_history, peak_y_history):
    # Update the plot of the current frame
    axs[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Current Frame")
    axs[0].axis('off')

    # Update the plot of the intensity distribution in the X direction
    axs[1].cla()
    axs[1].plot(intensity_x)
    axs[1].axvline(x=peak_x, color='r', linestyle='--')
    axs[1].set_title("Intensity Distribution in X Direction")
    axs[1].set_xlabel("Pixel")
    axs[1].set_ylabel("Intensity")

    # Update the plot of the intensity distribution in the Y direction
    axs[2].cla()
    axs[2].plot(intensity_y)
    axs[2].axvline(x=peak_y, color='r', linestyle='--')
    axs[2].set_title("Intensity Distribution in Y Direction")
    axs[2].set_xlabel("Pixel")
    axs[2].set_ylabel("Intensity")

    # Update the plot of the peak positions in the X direction
    axs[3].cla()
    axs[3].plot(peak_x_history, color='b')
    axs[3].set_title("Peak Position in X Direction")
    axs[3].set_xlabel("Frame")
    axs[3].set_ylabel("Position")

    # Update the plot of the peak positions in the Y direction
    axs[4].cla()
    axs[4].plot(peak_y_history, color='g')
    axs[4].set_title("Peak Position in Y Direction")
    axs[4].set_xlabel("Frame")
    axs[4].set_ylabel("Position")

    plt.draw()
    plt.pause(0.05)  # Short pause of 10ms to update the plot

# Initialize plot
def initialize_plot():
    plt.ion()
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))
    return fig, axs