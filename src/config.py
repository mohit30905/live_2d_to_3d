# Put camera intrinsics and basic config here. You should calibrate your phone camera to
# get fx,fy,px,py for your resolution. For quick testing, use these defaults for 640x480.
CONFIG = {
    "cam": {
        "width": 640,
        "height": 480,
        # These are example intrinsics; calibrate for your phone/crop.
        "fx": 768.0,
        "fy": 768.0,
        "px": 320.0,
        "py": 240.0
    },
    "depth": {
        # If you run calibration (see README), set depth_scale to convert MiDaS units -> meters
        "midas_scale": 1.0
    },
    "yolo": {
        "model": "yolov5s"  # change to yolov5n/yolov5s/yolov5l depending on speed
    }
}
