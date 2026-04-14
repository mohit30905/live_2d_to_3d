# Live 2D to 3D Object Reconstruction

A real-time computer vision pipeline that streams RGB frames from a mobile device, detects objects, estimates depth, and constructs isolated 3D point clouds. 

This project bridges 2D object detection and 3D spatial geometry, making it ideal for experimental 3D scanning, augmented reality applications, or spatial analysis.

## 🚀 Features

* **Real-Time IP Streaming:** Captures live RGB frames from any smartphone acting as an IP webcam.
* **Monocular Depth Estimation:** Uses the **MiDaS** model to generate depth maps from single-lens 2D images.
* **Object Detection:** Integrates **YOLOv5 / YOLOv8** to draw bounding boxes and isolate target objects within the frame.
* **3D Point Cloud Generation:** Leverages the pinhole camera model to project 2D pixels into 3D space.
* **Point Cloud Processing:** Utilizes depth-thresholding and **KMeans clustering** to clean the data and extract the distinct object from its background.
* **Export:** Outputs clean, processed `.ply` files alongside relevant metadata (JSON/PNG).

## 🛠️ Tech Stack

* **Python 3.x**
* **OpenCV** (Image processing and stream handling)
* **Open3D** (3D point cloud generation and manipulation)
* **PyTorch** (Model inference for MiDaS and YOLO)
* **Scikit-Learn** (KMeans clustering for noise reduction)

## 📁 Repository Structure

```text
live_2d_to_3d/
├── .vscode/               # Editor configurations
├── out/                   # Directory for generated outputs (.ply, .png, .json)
├── src/                   # Core source code (main.py, config.py, etc.)
├── requirement.txt        # Python dependencies
├── run_example.sh         # Shell script for quick execution
├── yolov5n.pt / yolov5s.pt / yolov8n.pt  # Pre-trained YOLO weights
└── README.md              # Project documentation