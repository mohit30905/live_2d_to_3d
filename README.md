This project uses a phone camera (as IP webcam) to stream RGB frames, estimates depth
using a monocular depth model (MiDaS), runs YOLOv5 for 2D detection, converts pixels
+ depth to 3D points (pinhole model), applies depth-thresholding and KMeans clustering
to isolate the object point cloud, and outputs a clean PLY + metadata.

See `src/` for code. Run the quick test after installing requirements.

Quick run (example):
1. Start an IP webcam app on your phone (e.g., IP Webcam / DroidCam / similar).
2. Note the video stream URL (e.g. http://192.168.0.12:8080/video).
3. Edit intrinsics in src/config.py to match your phone camera or use default.
4. Run: python src/main.py --stream-url http://192.168.0.12:8080/video

Notes about depth: Using MiDaS gives relative depth with scale ambiguity. The pipeline
includes optional calibration to convert relative depth to approximate meters.

