"""
ROI cropping, depth-threshold segmentation, and KMeans clustering to extract object point cloud.
"""
import numpy as np
from sklearn.cluster import KMeans
import open3d as o3d


def roi_points_from_depth(depth_map, bbox, intrinsics, depth_scale=1.0, invalid_value=0.0):
    x1,y1,x2,y2 = bbox
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(depth_map.shape[1]-1, x2); y2 = min(depth_map.shape[0]-1, y2)
    pts = []
    for v in range(y1, y2+1):
        for u in range(x1, x2+1):
            z = float(depth_map[v,u]) * depth_scale
            if z <= 0:
                continue
            p = np.array([ (u - intrinsics['px']) * z / intrinsics['fx'],
                           (v - intrinsics['py']) * z / intrinsics['fy'],
                           z ], dtype=np.float32)
            pts.append(p)
    if len(pts)==0:
        return np.zeros((0,3), dtype=np.float32)
    return np.vstack(pts)


def depth_threshold(pc, center_z, width_m):
    half = width_m
    mask = (pc[:,2] >= (center_z - half)) & (pc[:,2] <= (center_z + half))
    return pc[mask]


def kmeans_extract(pc, ref_point, k=2, random_state=0):
    if pc.shape[0] < k:
        return pc
    kmeans = KMeans(n_clusters=k, random_state=random_state).fit(pc)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    dists = ((centers - ref_point.reshape(1,3))**2).sum(axis=1)
    best = int(dists.argmin())
    return pc[labels == best]


def compute_sizes(pc):
    if pc.shape[0] == 0:
        return {}
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    aabb = pcd.get_axis_aligned_bounding_box()
    obb = pcd.get_oriented_bounding_box()
    return {
        'aabb_size': aabb.get_extent().tolist(),
        'obb_size': obb.extent.tolist(),
        'centroid': pcd.get_center()
    }


def visualize_pointcloud_with_boxes(points,
                                    pcd_info=None,
                                    show_mesh=False,
                                    poisson_depth=8,
                                    point_size=2.0,
                                    save_screenshot=None):
    """
    Visualize point cloud with AABB, OBB, centroid and optional Poisson mesh.

    Args:
        points (np.ndarray): Nx3 float array
        pcd_info (dict|None): result of compute_sizes() or None
        show_mesh (bool): run Poisson reconstruction and display mesh
        poisson_depth (int): Poisson depth parameter (8 default)
        point_size (float): size of points when rendering
        save_screenshot (str|None): path to save a PNG screenshot of view
    """
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError("Open3D is required for visualization. Install with `pip install open3d`") from e

    if points is None or points.shape[0] == 0:
        print("No points to visualize.")
        return

    # create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    # colorize by Z depth for nicer visualization
    pts = np.asarray(pcd.points)
    z = pts[:, 2]
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    # avoid division by zero
    if zmax - zmin < 1e-6:
        colors = np.tile(np.array([[0.6,0.6,0.6]]), (pts.shape[0],1))
    else:
        norm = (z - zmin) / (zmax - zmin)
        import matplotlib.cm as cm
        cmap = cm.get_cmap("viridis")
        colors = cmap(norm)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # create visual objects
    vis_geoms = [pcd]

    # AABB & OBB from Open3D (recompute from pcd to ensure consistency)
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1.0, 0.0, 0.0)  # red
    obb = pcd.get_oriented_bounding_box()
    obb.color = (0.0, 1.0, 0.0)   # green
    vis_geoms += [aabb, obb]

    # centroid sphere (use centroid from compute_sizes if provided)
    centroid = None
    if pcd_info is not None and 'centroid' in pcd_info:
        c = pcd_info['centroid']
        if isinstance(c, (list, tuple, np.ndarray)):
            centroid = np.asarray(c, dtype=np.float64)
        else:
            # if pcd.get_center() returned an o3d type, convert
            centroid = np.asarray(pcd.get_center(), dtype=np.float64)
    else:
        centroid = np.asarray(pcd.get_center(), dtype=np.float64)

    # small sphere at centroid
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=max(0.01, (aabb.get_extent().max() * 0.01)))
    sphere.translate(centroid)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([0.9, 0.75, 0.0])  # yellow-ish
    vis_geoms.append(sphere)

    # coordinate frame at centroid
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=aabb.get_extent().max() * 0.1)
    frame.translate(centroid)
    vis_geoms.append(frame)

    # optional Poisson mesh reconstruction (can be slow)
    mesh = None
    if show_mesh:
        print("Estimating normals for Poisson reconstruction...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=aabb.get_extent().max()*0.05, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(100)
        print("Running Poisson reconstruction (depth=%d)..." % poisson_depth)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)
        # crop mesh by AABB to remove far-off artifacts
        try:
            mesh_crop = mesh.crop(aabb)
            mesh_crop.compute_vertex_normals()
            mesh_crop.paint_uniform_color([0.7,0.7,0.7])
            vis_geoms.append(mesh_crop)
            mesh = mesh_crop
        except Exception:
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.7,0.7,0.7])
            vis_geoms.append(mesh)

    # Visualize with Open3D visualizer and optionally save screenshot
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PointCloud + Boxes", width=1280, height=720)
    for g in vis_geoms:
        vis.add_geometry(g)

    # Increase point size for visualization (works with render option)
    render_opt = vis.get_render_option()
    render_opt.point_size = float(point_size)
    render_opt.background_color = np.asarray([0.1, 0.1, 0.1])

    # set a reasonable viewpoint looking at centroid
    ctr = vis.get_view_control()
    bounds = aabb.get_center()
    ctr.set_lookat(bounds)
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.6)

    vis.run()

    # if save_screenshot is not None:
    #     # After run, capture screen (requires the window to be open)
    #     img = vis.capture_screen_float_buffer(do_render=True)
    #     import imageio
    #     imageio.imwrite(save_screenshot, (255 * np.asarray(img)).astype(np.uint8))
    #     print("Saved screenshot:", save_screenshot)

    vis.destroy_window()

    return mesh  # may be None
