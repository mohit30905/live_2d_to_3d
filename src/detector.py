from ultralytics import YOLO
import cv2
import time
from midas_depth import MiDaSDepth
import numpy as np
from config import CONFIG
import open3d as o3d
import background_removal as br


def detect():

    midas = MiDaSDepth('MiDaS_small') # instance of midas depth model

    model = YOLO("yolov8n.pt")  # or yolov8s/yolov8m etc.

    idx = 0 # 1 for usb cam 0 for pc cam 
    cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
    if not cap.isOpened():
        print("Failed to open index", idx)
    else:
        prev = 0
        while True:
            ret, frame = cap.read()
            depth_map = midas.predict(frame)
            
            cv2.imshow("Depth Map", depth_map)
            depth_map = filter_depth_range(depth_map, 0.35,0.85 )
            cv2.imshow("Filtered Depth Map", depth_map)
            print(depth_map)
            
            if not ret:
                break
            # ultralytics accepts BGR frames directly
            results = model(frame, stream=False)  # returns list-like Results
            # results[0].boxes.xyxy  etc.  (iterate boxes and draw)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # get b_obj
                    conf = float(box.conf[0])
                    cls = int(box.cls[0]) # generate C_obj
                    label = f"{model.names[cls]} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, label, (x1, y1-10) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
                    
                    # compute 2D center point
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    P2D_obj = (cx, cy)
                    # draw center point
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"P2D: {cx},{cy}", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            # fps
            now = time.time()
            fps = 1/(now-prev) if prev else 0
            prev = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            cv2.imshow("Ultralytics YOLO", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

            #background removal and point cloud extraction
            pts = br.roi_points_from_depth(depth_map, (100,100,400,400), intrinsics, depth_scale=5.0)
            print("ROI points shape:", pts.shape)
            pts = br.depth_threshold(pts, center_z=1.0, width_m=0.5)
            print("After depth threshold shape:", pts.shape)
            pts = br.kmeans_extract(pts, ref_point=np.array([0,0,1.0]), k=2)
            print("After KMeans shape:", pts.shape)
            sizes = br.compute_sizes(pts)
            print("Computed sizes:", sizes)


            points = dmtpc(depth_map , intrinsics , True , 5.0)
            print("Point cloud size:", points.shape)

            time.sleep(1)
            mesh = br.visualize_pointcloud_with_boxes(points, sizes, show_mesh=True, poisson_depth=8, point_size=2.0, save_screenshot="out/view.png")
            # optionally save the mesh
            if mesh is not None:
                o3d.io.write_triangle_mesh("out/mesh_poisson.ply", mesh)
                
            time.sleep(2)

    cap.release()
    cv2.destroyAllWindows()


def filter_depth_range(depth_map, min_depth, max_depth, replace_with=0.0):
    
    depth = depth_map.copy()

    # Create mask for valid range
    mask = (depth >= min_depth) & (depth <= max_depth)

    # Replace out-of-range pixels
    filtered = np.where(mask, depth, replace_with)

    return filtered.astype(np.float32)

def dmtpc(
    depth_map,
    intrinsics,
    depth_is_normalized=True,
    depth_to_meters_scale=5.0
):

    if depth_is_normalized:
        depth_m = depth_map.astype(np.float32) * depth_to_meters_scale
    else:
        depth_m = depth_map.astype(np.float32)

    # valid depth = non-zero
    mask = depth_m > 0
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((0,3), dtype=np.float32)

    zs = depth_m[ys, xs]

    fx = intrinsics['fx']
    fy = intrinsics['fy']
    px = intrinsics['px']
    py = intrinsics['py']

    # backproject
    Xs = (xs - px) * zs / fx
    Ys = (ys - py) * zs / fy
    Zs = zs

    pts = np.vstack((Xs, Ys, Zs)).T.astype(np.float32)
    return pts

def point_reconstriction(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

def mesh_reconstruction(pcd):
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        # 2. Poisson reconstruct
        pcd.orient_normals_consistent_tangent_plane(20)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        # 3. Crop mesh to region of interest
        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)
        # Show result
        o3d.visualization.draw_geometries([mesh])

    
intrinsics = CONFIG['cam']
detect()