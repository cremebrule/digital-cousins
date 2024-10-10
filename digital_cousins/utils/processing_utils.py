import numpy as np
import torch as th
import cv2
import json
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPoint
import digital_cousins.utils.transform_utils as T
import torch
from typing import List
from torchvision.ops import box_convert
import supervision as sv
import multiprocessing

def annotate(image_source: np.ndarray, boxes: torch.Tensor, phrases: List[str]) -> np.ndarray:
    """
    This function annotates an image with bounding boxes and labels.

    Args:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates.
    phrases (List[str] or None): A list of labels for each bounding box.
                            If None, phrases will not be drawn

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = phrases if phrases else None

    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame

def create_polygon_from_vertices(vertices):
    """
    Create a valid Polygon from a list of vertices projected onto the x-y plane.

    Args:
        vertices (list of tuples): A list of tuples representing the projected points on the x-y plane.

    Returns:
        None or Polygon: A Shapely Polygon object created from the vertices, or None if a valid polygon cannot be formed.
    """
    # Reduce to unique points, as the projection may result in duplicates
    unique_points = np.unique(vertices, axis=0)

    # Handle the special case of 4 unique points separately
    if len(unique_points) == 4:
        # Sorting points for a rectangle or a simple quadrilateral
        center = np.mean(unique_points, axis=0)
        angles = np.arctan2(unique_points[:,1] - center[1], unique_points[:,0] - center[0])
        sorted_points = unique_points[np.argsort(angles)]
        return Polygon(sorted_points)

    # For more than 4 points, use the convex hull to ensure a valid polygon
    # This works well for convex shapes but might not be ideal for concave ones
    if len(unique_points) > 4:
        convex_hull = MultiPoint(unique_points).convex_hull
        if isinstance(convex_hull, Polygon):
            return convex_hull
        else:
            raise AssertionError("Convex Hull Creation Fail")  # Convex hull is not a polygon (could be a line or a point)

    raise AssertionError("Not enough points for a polygon")  # Not enough points for a polygon

def project_vertices_to_plane(vertices, plane_coeff):
    """
    Project 3D vertices onto a plane using matrix operations and return 2D points (ignoring the z-axis).

    Args:
        vertices (np.array): Nx3 array of 3D vertices.
        plane_coeff (np.array): Plane equation coefficients [a, b, c, d].

    Returns:
        np.array: Nx2 array of 2D points (projected vertices) on the plane.
    """
    # Extract plane coefficients
    a, b, c, d = plane_coeff
    normal_vector = np.array([a, b, c])
    
    # Normalize the normal vector of the plane
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    
    # Project the vertices onto the plane
    distances = (np.dot(vertices, normal_vector) + d).reshape(-1, 1)
    projected_vertices = vertices - distances * normal_vector
    
    # Find two orthogonal vectors (v1 and v2) that form the local 2D coordinate system on the plane
    if np.abs(normal_vector[2]) > np.abs(normal_vector[0]):
        v1 = np.array([1, 0, -a/c])  # Create a vector orthogonal to the normal
    else:
        v1 = np.array([0, 1, -b/c])
    
    # Normalize v1 and compute v2
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal_vector, v1)
    
    # Construct a matrix that transforms the projected 3D points into 2D plane coordinates
    transformation_matrix = np.vstack([v1, v2]).T
    
    # Compute 2D coordinates by multiplying the projected vertices with the transformation matrix
    projected_2d_points = np.dot(projected_vertices, transformation_matrix)
    
    return projected_2d_points

def get_possible_obj_on_wall(wall_objs_info, is_lateral_wall=False, verbose=False, visualize=False):
    """
    Get all objects possible to be mounted on the wall.

    Args:
        vertices (List[dict]): A list of object info w.r.t. a wall, where each dictionary contains:
            object name, object aabb, the minimum and maximum distance between the object and the wall, 
            and a polygon formed by 8 vertices of the bounding box of the object projected on the wall plane.
        is_lateral_wall (bool): Whether the wall is a lateral wall, i.e., whose normal vector forms a large 
            angle with the camera's forward direction in the floor plane. When calling this function in ACDC,
            wall angle > 45 degree is regarded as a lateral wall.
        verbose (bool): Whether to print out progress.
        visualize (bool): Whether to visualize projected bounding boxes on a wall plane.

    Returns:
        Set: A set of all objects possible to be mounted on the wall.
    """
    all_obj_names = {info["name"] for info in wall_objs_info}
    invalid_obj_names = set()
    wall_objs_info.sort(key=lambda x: x["wall_dist_min"])    # Sort objs in terms of their minimum distance to the wall
    for i in range(len(wall_objs_info)-1):
        obj1_info = wall_objs_info[i]
        for j in range(i+1, len(wall_objs_info)):
            obj2_info = wall_objs_info[j]
            if obj2_info["name"] in invalid_obj_names:
                continue
            
            if visualize:
                def vis():
                    fig, ax = plt.subplots()

                    # Get the coordinates of the first polygon
                    x1, y1 = obj1_info["polygon"].exterior.xy
                    ax.plot(x1, y1, color='blue', label=obj1_info['name'])

                    # Get the coordinates of the second polygon
                    x2, y2 = obj2_info["polygon"].exterior.xy
                    ax.plot(x2, y2, color='green', label=obj2_info['name'])

                    # Add legend, labels, and title
                    ax.legend()
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_title('Visualization of Projected Bounding Box')

                    # Display the plot in a pop-up window
                    plt.show()
                vis_process = multiprocessing.Process(target=vis)
                vis_process.start()
                vis_process.join()

            if obj1_info["polygon"].intersects(obj2_info["polygon"]):
                intersect_area = obj1_info["polygon"].intersection(obj2_info["polygon"]).area
                # If the intersection area exceeds 20% area of either polygon, 
                # the obj with larger min dist is not possible to be mounted on the wall
                if intersect_area / obj1_info["polygon"].area >= 0.33 or intersect_area / obj2_info["polygon"].area >= 0.33:
                    if verbose:
                        print(f"{obj1_info['name']} intersect with {obj2_info['name']}. Add {obj2_info['name']} as invalid")
                    invalid_obj_names.add(obj2_info["name"])
                    continue
            
            if is_lateral_wall: 
                # For lateral wall, if resizing an object causes shrinkage bounding box of another object with smaller distance to a wall,
                # or unreasonably large bounding box of itself, it is impossible for the object to make physical contact with the wall.
                obj1_bottom_height = min(obj1_info["vertices"][0][-1], obj1_info["vertices"][1][-1])
                obj2_bottom_height = min(obj2_info["vertices"][0][-1], obj2_info["vertices"][1][-1])
                obj1_center_height = (obj1_info["vertices"][0][-1] + obj1_info["vertices"][1][-1]) / 2
                obj2_center_height = (obj2_info["vertices"][0][-1] + obj2_info["vertices"][1][-1]) / 2
                obj1_top_height = max(obj1_info["vertices"][0][-1], obj1_info["vertices"][1][-1])
                obj2_top_height = max(obj2_info["vertices"][0][-1], obj2_info["vertices"][1][-1])
                if ((obj1_bottom_height <= obj2_center_height and obj1_top_height >= obj2_bottom_height) or (obj2_bottom_height <= obj1_center_height and obj2_top_height >= obj1_bottom_height)) \
                    and obj1_info["wall_dist_max"] < obj2_info["wall_dist_min"]:
                    if verbose:
                        print(f"Resizing {obj2_info['name']} can cause shrinkage of bounding box of {obj1_info['name']}, or unreasonably large bounding box of {obj2_info['name']}. Add {obj2_info['name']} as invalid")
                    invalid_obj_names.add(obj2_info["name"])

    return all_obj_names - invalid_obj_names

def rescale_image(img, in_limits, out_limits):
    """
    Rescales image from having values in range @in_limits to range @out_limits

    Args:
        img (np.ndarray): Image to be rescaled
        in_limits (2-tuple): (min, max) range of the input image
        out_limits (2-tuple): (min, max) range of the rescaled image

    Returns:
        np.ndarray: Rescaled image
    """
    # Out shape if specified should be (H, W) 2-tuple
    # Keep absolute range to be from 0 to 10 --> renormalize to 0 - 1
    in_min, in_max = in_limits
    out_min, out_max = out_limits
    scale_factor = abs(out_max - out_min) / abs(in_max - in_min)
    out_tf = (out_max + out_min) / 2.0
    in_tf = (in_max + in_min) / 2.0
    img = (img - in_tf) * scale_factor + out_tf
    return img



def process_depth_linear(depth, in_limits=(0, 10), out_limits=(0, 1), out_shape=None, use_16bit=True):
    """
    Discretizes a linear depth map from range @in_limits to range @out_limits, and optionally resizes it to @out_shape

    Args:
        depth (np.ndarray): Input depth map with metric values
        in_limits (2-tuple): (min, max) range of the input image
        out_limits (2-tuple): (min, max) range of the processed image
        out_shape (None or 2-tuple): If specified, the (H, W) @depth should be resized to
        use_16bit (bool): Whether to use 16-bit or 8-bit uint encoding when rescaling the image

    Returns:
        np.ndarray: Processed depth image.
    """
    # Out shape if specified should be (H, W) 2-tuple
    # Keep absolute range to be from 0 to 10 --> renormalize to 0 - 1
    in_min, in_max = in_limits
    foreground = np.where(depth <= in_max)
    background = np.where(depth > in_max)
    depth = rescale_image(img=depth, in_limits=in_limits, out_limits=out_limits)

    # Zero out background
    bit_size = 16 if use_16bit else 8
    dtype = np.uint16 if use_16bit else np.uint8
    depth[background] = 0.0
    # Multiply by 2 ** 16
    depth = depth * (2 ** bit_size)
    depth = depth.astype(dtype)
    if out_shape is not None:
        depth = cv2.resize(depth, (out_shape[1], out_shape[0]))
    return depth


def unprocess_depth_linear(depth, in_limits=(0.0, 1.0), out_limits=(0.0, 10.0)):
    """
    Unnormalizes a linear depth map from range @in_limits to range @out_limits. This process is the inverse
    of @procedss_depth_linear)

    Args:
        depth (np.ndarray): Input depth map with normalized values (the output of @process_depth_linear)
        in_limits (2-tuple): (min, max) range of the input image
        out_limits (2-tuple): (min, max) range of the unprocessed image
    
    Returns:
        np.ndarray: Unnormalized depth image.
    """
    # Map to float, divide by 2**16, then invert transform
    depth = depth.astype(float)
    depth = depth / (2 ** 16)

    # Keep unnormalize from 0 - 1 --> 0 - 10
    in_min, in_max = in_limits
    out_min, out_max = out_limits
    scale_factor = abs(out_max - out_min) / abs(in_max - in_min)
    out_tf = (out_max + out_min) / 2.0
    in_tf = (in_max + in_min) / 2.0
    depth = (depth - in_tf) * scale_factor + out_tf
    return depth

def compute_point_cloud_from_depth(depth, K, cam_to_img_tf=None, world_to_cam_tf=None, visualize_every=0,
                                   grid_limits=None):
    """
    Computes point cloud from depth image.

    Args:
        depth (np.ndarray): Input depth map with normalized values (the output of @process_depth_linear)
        K (np.ndarray): 3x3 cam intrinsics matrix
        cam_to_img_tf (np.ndarray): 4x4 Camera to image coordinate transformation matrix.
                    omni cam_to_img_tf is T.pose2mat(([0, 0, 0], T.euler2quat([np.pi, 0, 0])))
        world_to_cam_tf (np.ndarray): 4x4 World to camera coordinate transformation matrix.
        visualize_every (int): Step size when uniformly sampling points in the resulting point cloud to visualize.
        grid_limits (float): Visualization plot grid limits.
    
    Returns:
        np.ndarray: Resulting point cloud.
    """
    
    h, w = depth.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij", sparse=False)
    assert depth.min() >= 0
    u = x
    v = y
    uv = np.dstack((u, v, np.ones_like(u)))

    Kinv = np.linalg.inv(K)

    pc = depth.reshape(-1, 1) * (uv.reshape(-1, 3) @ Kinv.T)
    pc = pc.reshape(h, w, 3)

    # If no tfs, use identity matrix
    cam_to_img_tf = np.eye(4) if cam_to_img_tf is None else cam_to_img_tf
    world_to_cam_tf = np.eye(4) if world_to_cam_tf is None else world_to_cam_tf

    pc = np.concatenate([pc.reshape(-1, 3), np.ones((h * w, 1))], axis=-1)  # shape (H*W, 4)

    # Convert using camera transform
    # Create (H * W, 4) vector from pc
    pc = (pc @ cam_to_img_tf.T @ world_to_cam_tf.T)[:, :3].reshape(h, w, 3)

    if visualize_every > 0:
        def vis():
            pc_flat = np.array(pc.reshape(-1, 3))
            pc_flat = np.where(np.linalg.norm(pc_flat, axis=-1, keepdims=True) > 1e4, 0.0, pc_flat)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(pc_flat[::visualize_every, 0], pc_flat[::visualize_every, 1], pc_flat[::visualize_every, 2], s=1)
            if grid_limits is not None:
                ax.set_xbound(lower=-grid_limits, upper=grid_limits)
                ax.set_ybound(lower=-grid_limits, upper=grid_limits)
                ax.set_zbound(lower=-grid_limits, upper=grid_limits)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()
        vis_process = multiprocessing.Process(target=vis)
        vis_process.start()
        vis_process.join()

    return pc


def compute_bbox_from_mask(mask_fpath):
    mask = Image.open(mask_fpath)
    W, H = mask.size
    segmentation = np.where(np.array(mask))
    x_min = int(np.min(segmentation[1]))
    x_max = int(np.max(segmentation[1]))
    y_min = int(np.min(segmentation[0]))
    y_max = int(np.max(segmentation[0]))
    bbox = x_min, y_min, x_max, y_max
    bbox = np.array(bbox) / np.array([W, H, W, H])
    return bbox


def load_mask(mask_dir):
    """Load a mask image and convert it to a binary numpy array."""
    mask = Image.open(mask_dir).convert('1')  # Convert to binary image
    return np.array(mask, dtype=np.uint8)


def mask_area(mask):
    """Calculate the area of the mask."""
    return np.sum(mask)


def mask_intersection_area(mask1, mask2):
    """Calculate the intersection area between two masks."""
    intersection = np.logical_and(mask1, mask2)
    return np.sum(intersection)


def filter_large_masks(all_mask_dirs, prop_area_threshold=0.7):
    """Filter out large masks that contain smaller masks."""
    masks = [(load_mask(mask_dir), mask_dir) for mask_dir in all_mask_dirs]
    remaining_masks = []

    for i, (mask1, dir1) in enumerate(masks):
        is_larger_mask = False
        for j, (mask2, dir2) in enumerate(masks):
            if i != j:
                inter_area = mask_intersection_area(mask1, mask2)
                min_area = min(mask_area(mask1), mask_area(mask2))

                if inter_area > prop_area_threshold * min_area:
                    # If mask1 is the larger one, we mark it for removal
                    if mask_area(mask1) > mask_area(mask2):
                        is_larger_mask = True
                        break

        if not is_larger_mask:
            remaining_masks.append(dir1)

    return remaining_masks


def shrink_mask(mask, kernel_size=(20, 20), iterations=10):
    """
    Shrinks a binary mask by performing morphological erosion.

    Args:
        mask (numpy.ndarray): A 2D numpy array with boolean values (True/False), where True indicates floor.
        kernel_size (tuple of int): Kernel size used to perform binary erosion. Larger kernel size will lead to more aggresize erosion
        iterations (int): Number of times the erosion operation is applied.

    Returns:
        numpy.ndarray: A shrunk version of the input mask.
    """
    from scipy.ndimage import binary_erosion

    # Define the structure for erosion (use a 3x3 square by default)
    structure = np.ones(kernel_size, dtype=bool)

    # Perform erosion
    shrunk_mask = binary_erosion(mask, structure=structure, iterations=iterations)

    return shrunk_mask


def display_inlier_outlier(cloud, ind):
    """
    Visualize inliers (gray) and outliers (red) of a point cloud with different colors.

    Args:
        cloud (Open3d.geometry.PointCloud): Object point cloud.
        ind (List): Indices of inliers in cloud.
    """
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    vis_cloud_list = []
    if len(inlier_cloud.points) == 0:
        print(f"No inliers")
    else:
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        vis_cloud_list.append(inlier_cloud)

    if len(outlier_cloud.points) == 0:
        print(f"No outliers")
    else:
        outlier_cloud.paint_uniform_color([1, 0, 0])
        vis_cloud_list.append(outlier_cloud)

    if vis_cloud_list:
        def vis():
            o3d.visualization.draw_geometries(vis_cloud_list)
        vis_process = multiprocessing.Process(target=vis)
        vis_process.start()
        vis_process.join()

def denoise_obj_point_cloud(
        pcd_obj, 
        visualize_process=False,
        visualize_result=False,
        thred_percentage=0.05,
        absolute_threshold=200,
        large_pc_threshold=12000,
        noise_thred_percentage = 0.65,
        noise_thred_percentage_large_pc = 0.4,
    ):
    """
    Removes noise from object point by a set of point cloud denoising methods.
    Default values for arguments are tuned for resized images with longer side equals 1600 pixels.

    Args:
        pcd_obj (open3d.geometry.PointCloud): Object point cloud.
        visualize_process (bool): Whether to visualize the denoising process
        visualize_result (bool): Whether to visualize denoising result of the downsampled point cloud
        thred_percentage (float): Clusters with points more than total points number * thred_percentage
                            will not be classified as noise clusters.
        absolute_threshold (float): Clusters with points more than absolute_threshold will not be 
                            classified as noise clusters. This check will be skipped for large point clouds.
        large_pc_threshold (int): Object point clouds with more than @large_pc_threshold points will be 
                            denoised as large object point cloud.
        noise_thred_percentage (float): If the noise cluster (the cluster with label -1) has more than 
                            total points number * thred_percentage, the noise cluster will not be removed
        noise_thred_percentage_large_pc (float): For large point cloud, if the noise cluster (the cluster with label -1) 
                            has more than total points number * thred_percentage, the noise cluster will not be removed

    Returns:
        open3d.geometry.PointCloud: Denoised object point cloud.
        list: Corresponding point indices in pcd_obj.
    """

    if visualize_process:
        def vis():
            o3d.visualization.draw_geometries([pcd_obj])
        vis_process = multiprocessing.Process(target=vis)
        vis_process.start()
        vis_process.join()

    # 1. Downsample the object point cloud
    # Downsample more aggresively for object point clouds with more points
    downsample_factor = min(4, 1 + len(pcd_obj.points) // 1000)                                         
    downsample_factor = 5 if len(pcd_obj.points) > large_pc_threshold else downsample_factor
    downsampled_indices = np.arange(len(pcd_obj.points))[::downsample_factor]
    pcd_obj_downsampled = pcd_obj.select_by_index(downsampled_indices)
    if visualize_process:
        def vis():
            o3d.visualization.draw_geometries([pcd_obj_downsampled])
        vis_process = multiprocessing.Process(target=vis)
        vis_process.start()
        vis_process.join()

    # 2. Apply out-of-the-shelf statistical outlier removal
    pcd_obj_downsampled, ind = pcd_obj_downsampled.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Retain only the valid downsampled indices after noise removal
    valid_downsampled_indices = downsampled_indices[ind]

    # 3. Perform DBSCAN clustering
    min_points_cluster = min(max(8, len(pcd_obj_downsampled.points) // 1500 * 5), 25)
    eps_cluster = min(0.05, max(0.012, (len(pcd_obj.points) / 4000 + 2) * 0.005)) # Use smaller eps for smaller point cloud
    labels = np.array(pcd_obj_downsampled.cluster_dbscan(eps=eps_cluster, min_points=min_points_cluster, print_progress=False))
    max_label = labels.max()
    original_color = np.asarray(pcd_obj_downsampled.colors).copy()

    if visualize_process:
        def vis():
            cmap = plt.get_cmap("tab20")
            colors = cmap(labels / (max_label if max_label > 0 else 1))
            colors[labels < 0] = 0
            pcd_obj_downsampled.colors = o3d.utility.Vector3dVector(colors[:, :3])
            o3d.visualization.draw_geometries([pcd_obj_downsampled])
        vis_process = multiprocessing.Process(target=vis)
        vis_process.start()
        vis_process.join()
    
    # Identify unique clusters and count the number of points in each cluster
    total_points = len(labels)
    unique_clusters, counts = np.unique(labels, return_counts=True)

    # List to store the indices of clusters considered as noise
    noise_clusters = []

    # Threshold for determining if a cluster is significant (not noise)
    significant_threshold = thred_percentage * total_points
    
    # 4. Remove non-significant clusters where most of points are outside the bbox fitted by other clusters
    clean = False
    while not clean:
        clean = True
        for cluster_id, count in zip(unique_clusters, counts):
            if cluster_id == -1 and cluster_id not in noise_clusters and \
                ((total_points < large_pc_threshold and count < noise_thred_percentage * total_points) or \
                (total_points >= large_pc_threshold and noise_thred_percentage_large_pc)):
                noise_clusters.append(cluster_id)
                clean = False
                significant_threshold = thred_percentage * (total_points - count)
                continue
            if cluster_id in noise_clusters or count > significant_threshold or (total_points < large_pc_threshold and count > absolute_threshold):
                continue
            
            # For noise candidates, determine if they should be removed
            cluster_indices = np.where(labels == cluster_id)[0]

            # Exclude points with cluster_id in noise_clusters or the current cluster_id
            exclude_clusters = np.append(noise_clusters, cluster_id)
            non_noise_indices = np.where(~np.isin(labels, exclude_clusters))[0]
            
            if len(non_noise_indices) == 0:
                continue  # No non-noise data to compare against
            
            # Compute bounding box of the non-noise points
            non_noise_pcd = pcd_obj_downsampled.select_by_index(non_noise_indices)
            non_noise_pcd.colors = o3d.utility.Vector3dVector(np.repeat([[1, 0, 0]], len(non_noise_pcd.points), axis=0).astype(np.float64))
            bbox = non_noise_pcd.get_oriented_bounding_box()
            
            # Check points of the current cluster against the bounding box
            cluster_points = pcd_obj_downsampled.select_by_index(cluster_indices)
            bbox.color = (0,1,0)

            # Get indices of points that are inside the oriented bounding box
            cluster_points_array = np.asarray(cluster_points.points)
            cluster_points_o3d = o3d.utility.Vector3dVector(cluster_points_array)
            inside_indices = bbox.get_point_indices_within_bounding_box(cluster_points_o3d)

            # Count points outside the oriented bounding box
            outside_count = len(cluster_points.points) - len(inside_indices)
            
            # If more than 70% of the points are outside the bounding box, remove this cluster
            if outside_count > 0.7 * count:
                noise_clusters.append(cluster_id)
                clean = False

    # Removing noise clusters
    valid_clusters = [cluster_id for cluster_id in unique_clusters if cluster_id not in noise_clusters]
    valid_indices_in_downsampled = [i for i in range(total_points) if labels[i] in valid_clusters]

    # Get the final valid indices in the original point cloud by mapping back the downsampled indices
    valid_indices_in_original = valid_downsampled_indices[valid_indices_in_downsampled]
    
    # Create a cleaned point cloud from valid points
    cleaned_pcd = pcd_obj_downsampled.select_by_index(valid_indices_in_downsampled)
    
    # Restore the original colors for the cleaned point cloud
    cleaned_pcd.colors = o3d.utility.Vector3dVector(original_color[valid_indices_in_downsampled])

    if visualize_result:
        display_inlier_outlier(pcd_obj_downsampled, valid_indices_in_downsampled)
        def vis():
            aligned_bbox = cleaned_pcd.get_axis_aligned_bounding_box()
            aligned_bbox.color = (1, 0, 0)
            oriented_bbox = cleaned_pcd.get_oriented_bounding_box()
            oriented_bbox.color = (0, 1, 0)
            o3d.visualization.draw_geometries([cleaned_pcd, aligned_bbox, oriented_bbox])
        vis_process = multiprocessing.Process(target=vis)
        vis_process.start()
        vis_process.join()

    # Return the cleaned point cloud
    return cleaned_pcd, valid_indices_in_original

def get_aabb_vertices(aabb):
    """
    Get coordinates of 8 vertices of 3D axis-aligned bounding box.

    Args:
        aabb (open3d.geometry.AxisAlignedBoundingBox): Object point cloud.

    Returns:
        list: Coordinates of 8 vertices.
    """
    min_bound = aabb.min_bound
    max_bound = aabb.max_bound
    
    # Generate the 8 vertices of the AABB
    vertices = [[min_bound[0], min_bound[1], min_bound[2]],
                [min_bound[0], min_bound[1], max_bound[2]],
                [min_bound[0], max_bound[1], min_bound[2]],
                [min_bound[0], max_bound[1], max_bound[2]],
                [max_bound[0], min_bound[1], min_bound[2]],
                [max_bound[0], min_bound[1], max_bound[2]],
                [max_bound[0], max_bound[1], min_bound[2]],
                [max_bound[0], max_bound[1], max_bound[2]]]
    return vertices


def remove_component(vec, dir_to_remove, normalize=False):
    """
    Removes component from @vec, optionally normalizing if requested

    Args:
        vec (np.ndarray): (x,y,z) vector to project onto subspace
        dir_to_remove (np.ndarray): (x,y,z) vector whose direction should be removed from @vec
        normalize (bool): Whether to normalize the resulting vector or not
    """
    assert np.isclose(np.linalg.norm(dir_to_remove), 1.0)
    vec_dir_proj = np.dot(vec, dir_to_remove) * dir_to_remove
    vec_dir_perp_proj = vec - vec_dir_proj
    return vec_dir_perp_proj / np.linalg.norm(vec_dir_perp_proj) if normalize else vec_dir_perp_proj


def points_in_same_direction(vec_a, vec_b):
    """
    Determines whether @vec_a and @vec_b point in the same direction or not

    Args:
        vec_a (np.ndarray): (x,y,z) vector A
        vec_b (np.ndarray): (x,y,z) vector B

    Returns:
        bool: Whether both vectors point in same direction or not
    """
    return np.dot(vec_a / np.linalg.norm(vec_a), vec_b / np.linalg.norm(vec_b)) > 0


def get_reproject_offset(
    pc_obj,
    z_dir,
    xy_dist,    # taken wrt z_dir direction
    z_dist     # taken wrt plane perpendicular from z_dir direction
):
    """
    Reprojects an image from a given dataset into its equivalent pan and tilt angle offset with respect to the
    camera pose

    Args:
        pc_obj (np.ndarray): (N, 3) object point cloud, specified in camera A's frame
        z_dir (np.ndarray): (x,y,z) direction of the global z-axis, specified in camera A's frame
        xy_dist (float): Horizontal distance from the object bbox center to camera B's frame
        z_dist (float): Vertical distance from the object bbox center to camera B's frame

    Returns:
        2-tuple:
            - float: Pan angle offset
            - float: Tilt angle offset
    """

    # Get bbox center of obj
    obj_min, obj_max = pc_obj.min(axis=0), pc_obj.max(axis=0)
    obj_pos = (obj_min + obj_max) / 2.0

    # Get biggest dimension
    scale_factor = np.max(obj_max - obj_min)

    # In OG, our relative position wrt the object's bbox is [0, -2.30, 0.65] given the maximum width is 1. So, we
    # similar scale our distance proportional to the corresponding scale factor
    # So want our camera facing this object pose, with relative pose 0.65 in the computed z_direction,
    # and -2.30 in the direction of parallel to the ground plane and pointing in the direction of the camera

    # compute which way is "up" with respect to the image frame
    cam_to_obj = obj_pos
    cam_to_obj_z_perp_dir = remove_component(vec=cam_to_obj, dir_to_remove=z_dir, normalize=True)
    new_cam_pos = obj_pos + z_dist * scale_factor * z_dir - xy_dist * scale_factor * cam_to_obj_z_perp_dir
    tilt_dir = np.cross(z_dir, cam_to_obj_z_perp_dir)

    # The resulting orientation tf will be the tf to get from direction vector [0, 0, 1] to direction
    # obj_pos - new_cam_pos, with the assumption there is zero roll (i.e.: camera is not tilted wrt the ground)
    # This is a compounding of two angles:
    # (a) pan angle, which is derived from cosine between projection of current cam direction [0, 0, 1] and desired
    #     cam direction onto the plane defined by z_dir, and
    # (b) tilt angle, which is derived from the cosine between projection of current cam direction [0, 0, 1] and desired
    #     cam direction onto the plane defined by tilt_dir
    cur_cam_dir = np.array([0, 0, 1.0])
    new_cam_dir = obj_pos - new_cam_pos
    cur_cam_pan_proj = remove_component(vec=cur_cam_dir, dir_to_remove=z_dir, normalize=True)
    new_cam_pan_proj = remove_component(vec=new_cam_dir, dir_to_remove=z_dir, normalize=True)
    pan_angle_dir = 1 if points_in_same_direction(np.cross(cur_cam_pan_proj, new_cam_pan_proj), z_dir) else -1
    pan_angle = np.arccos(np.dot(cur_cam_pan_proj, new_cam_pan_proj)) * pan_angle_dir
    pan_tf = T.axisangle2mat(pan_angle * z_dir)
    cur_cam_tilt_proj = remove_component(vec=cur_cam_dir, dir_to_remove=tilt_dir, normalize=True)
    new_cam_tilt_proj = remove_component(vec=new_cam_dir, dir_to_remove=tilt_dir, normalize=True)
    tilt_angle_dir = 1 if points_in_same_direction(np.cross(cur_cam_tilt_proj, new_cam_tilt_proj), z_dir) else -1
    tilt_angle = np.arccos(np.dot(cur_cam_tilt_proj, new_cam_tilt_proj)) * tilt_angle_dir
    tilt_tf = T.axisangle2mat(tilt_angle * tilt_dir)

    # Compose final 4x4 tf
    cam_tf = np.eye(4)
    cam_tf[:3, 3] = new_cam_pos
    cam_tf[:3, :3] = tilt_tf @ pan_tf

    return pan_angle, tilt_angle


def distance_to_plane(point, plane_params, keep_sign=False):
    x1, y1, z1 = point
    a, b, c, d = plane_params
    if keep_sign:
        # Allow distance between points behind the plane and the plane to be negative
        numerator = a*x1 + b*y1 + c*z1 + d
    else:
        numerator = abs(a*x1 + b*y1 + c*z1 + d)
    denominator = np.sqrt(a**2 + b**2 + c**2)
    distance = numerator / denominator
    return distance


def is_overlapped_box(box1_bottom_left, box1_upper_right, box2_bottom_left, box2_upper_right):
    return not (box1_upper_right[0] < box2_bottom_left[0]
                or box1_bottom_left[0] > box2_upper_right[0]
                or box1_upper_right[1] < box2_bottom_left[1]
                or box1_bottom_left[1] > box2_upper_right[1])


def resize_image(img, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Rescales image @img, maintaining its aspect ratio.
    Source taken from: https://stackoverflow.com/a/44659589

    Args:
        img (np.ndarray): (H,W,3) Image to resize
        width (None or int): If specified, new width (in pixels) to resize image to. Only one of @width, @height should
            be specified
        height (None or int): If specified, new height (in pixels) to resize image to. Only one of @width, @height
            should be specified
        inter (int): Type of interpolation to use
    """
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = img.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return img

    # check to see if the width is None
    assert width is None or height is None, "Only one of @width and @height should be specified!"

    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(img, dim, interpolation=inter)

    # return the resized image
    return resized


class NumpyTorchEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, th.Tensor):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
