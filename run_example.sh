# Example run script
# Start your phone IP webcam app, get stream URL, then edit the URL below and run this script.
python src/main.py --stream-url "http://192.168.0.12:8080/video" --out results --max-frames 200


### Notes for the semester project structure
# - README.md explains setup and calibration.
# - requirements.txt lists dependencies.
# - src/ contains modular code: capture, detection, depth, mapping, removal, utils
# - results/ will contain .ply and .json files saved during runs
# - You can extend the project by adding hand-pose estimation and SG-GM mapping modules.

# End of canvas file