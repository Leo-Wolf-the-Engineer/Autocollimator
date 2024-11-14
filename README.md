# Autocollimator Software

Using a Basler GigE Webcam
The Live view:
![image](images/autocollimator_live_20241113_210404.png)
The Straightness measuring interface:
![image](images/straightness_measurement_20241113_213029.png)
Autocollimator WIP 
This is the first crappy setup. It delivers surprisingly good results. It can hold within 2 arcseconds for several minutes (angle over time in both X and Y are the two plots on the right). It is very sensitive to vibration (even loud music?).
Just using a crappy laser diode, a beam splitter(without anti reflection coating :/), a 385mm lens and a Basler Camera.

With a ton of help from AI I got the Python Programm running within a single day. It's on my GitHub page. It averages the intensity of the projected circle to find the center (rather poorly atm).
The camera can handle up to 1500 frames per second at full HD. This allows for tons of averaging, which also further reduces measurement noise.
I also implemented a straightness measurement procedure already. Here measurements over a certain amount of time (typically 3s) are averaged at each position. I added boxes to be able to enter various parameters. It directly depends the values and calculates the peak-to-valley deviation. 

There is a lot left to do but this is very promising. One important thing is building a sine bar to confirm the measurements.

## Installation

- Clone the repository to your system
- Install python 3.10 or newer: [get python](https://www.python.org/downloads/)
- Install pip package manager with ```python -m ensurepip --upgrade```
- Install requirements with ```pip install -r requirements.txt```
