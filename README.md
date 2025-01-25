# Virtual Camera Effects

A Python application that creates a virtual camera with real-time effects including:
- Background blur (keeps your body in focus while blurring the background)
- Face censoring toggle (pixelates and blurs your face)
- Hand tracking visualization with real-time measurements
- Hand skeleton visualization with distance measurements in centimeters

## Features

- **Background Blur**: Automatically detects your body and blurs everything else
- **Face Censoring**: Toggle face pixelation/blur with the 'N' key
- **Hand Tracking**: Shows hand skeleton visualization with real-time measurements
- **Distance Measurements**: Toggle display of segment lengths in centimeters for each part of the hand
- **Virtual Camera**: Works as a virtual camera device for use in Discord, Zoom, Teams, etc.

## Prerequisites

Before running this application, you need to install a virtual camera driver:

### Windows
1. Install [OBS Studio](https://obsproject.com/) which includes the virtual camera driver
2. Run OBS once to install the virtual camera

### Linux
```bash
sudo apt-get install v4l2loopback-dkms
```

### macOS
1. Install [OBS Studio](https://obsproject.com/) which includes the virtual camera driver
2. Run OBS once to install the virtual camera

## Installation

1. Clone this repository:
```bash
git clone [your-repo-url]
cd [your-repo-name]
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the program:
```bash
python main.py
```

2. Controls:
   - Press 'N' to toggle face censoring on/off
   - Press 'M' to toggle hand measurements on/off
   - Press 'Q' to quit the program

3. To use in other applications:
   - Open your preferred video chat application (Discord, Zoom, etc.)
   - In the camera settings, select "OBS Virtual Camera" (or similar name)
   - The effects will now be visible in your video feed

## How it Works

The application uses several computer vision technologies:
- MediaPipe for body segmentation, hand tracking, and face detection
- OpenCV for image processing and camera handling
- PyVirtualCam for virtual camera output

## Requirements

- Python 3.8 or higher
- OpenCV
- MediaPipe
- NumPy
- PyVirtualCam
- Virtual camera driver (OBS Virtual Camera or v4l2loopback)

## Troubleshooting

1. **No Virtual Camera Available**
   - Make sure you've installed OBS Studio (Windows/macOS) or v4l2loopback (Linux)
   - Try running OBS once to initialize the virtual camera

2. **Performance Issues**
   - Close other applications using your camera
   - Ensure your computer meets the minimum requirements for running real-time computer vision applications

3. **Camera Not Found**
   - Check if other applications are using your camera
   - Try unplugging and reconnecting your camera
   - Verify camera permissions for Python/your IDE

## License

This project is open source and free to use. You can:
- Use the code for any purpose
- Study how the code works and modify it
- Redistribute copies of the original or modified code
- Distribute the code commercially

There are no restrictions on using, modifying, or distributing this code. Feel free to use it in your own projects!

## Acknowledgments

This project uses several open-source libraries:
- [MediaPipe](https://mediapipe.dev/)
- [OpenCV](https://opencv.org/)
- [PyVirtualCam](https://github.com/letmaik/pyvirtualcam) 
