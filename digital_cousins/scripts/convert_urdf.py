import omnigibson as og
from omnigibson.scenes import Scene
from omnigibson.objects.dataset_object import DatasetObject, DatasetType
from omnigibson.utils.sampling_utils import raytest_batch
import omnigibson.utils.transform_utils_np as T
import os
import numpy as np
import copy
import xml.etree.ElementTree as ET
from xml.dom import minidom
import shutil
from omnigibson.macros import gm

class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        else:
            self.parent[root_y] = root_x
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_x] += 1

def perform_raycast(start_points, end_points, only_closest=True):
    """Wrapper function for raycasting to handle errors gracefully"""
    try:
        return raytest_batch(start_points=start_points, end_points=end_points, only_closest=only_closest)
    except Exception as e:
        if verbose:
            print(f"Error during ray casting: {e}")
        return []

def log(message, level=1):
    """Log messages based on verbosity level"""
    if verbose >= level:
        print(message)

def determine_base_mesh(obj):
    """Step 1: Determine base mesh by finding the mesh with largest AABB volume"""
    log("==================== Step1: Determine Base Mesh ====================")
    largest_volume = 0
    largest_link_name = None
    
    for link_name, prim in obj.links.items():
        if link_name == "base_link":
            continue
        
        link_extent = prim.aabb_extent
        link_volume = float(link_extent[0]) * float(link_extent[1]) * float(link_extent[2])
        
        if link_volume > largest_volume:
            largest_volume = link_volume
            largest_link_name = link_name
    
    log(f"Base link identified as: {largest_link_name}")
    return largest_link_name

def specify_front_direction(obj, base_link_name, sampling_width=0.0010, sampling_offset=1.0, max_rays=5000):
    """Step 2: Specify the front direction of the object mesh by dense ray casting"""
    log("==================== Step2: Specify Front Direction ====================")
    obj_aabb_min, obj_aabb_max = obj.aabb
    obj_aabb_center = obj.aabb_center
    obj_aabb_extent = obj.aabb_extent
    
    directions = [
        {"name": "+X", "offset_vec": [sampling_offset, 0, 0], "ray_vec": [-sampling_offset - obj_aabb_extent[0], 0, 0]},
        {"name": "-X", "offset_vec": [-sampling_offset, 0, 0], "ray_vec": [sampling_offset + obj_aabb_extent[0], 0, 0]},
        {"name": "+Y", "offset_vec": [0, sampling_offset, 0], "ray_vec": [0, -sampling_offset - obj_aabb_extent[1], 0]},
        {"name": "-Y", "offset_vec": [0, -sampling_offset, 0], "ray_vec": [0, sampling_offset + obj_aabb_extent[1], 0]},
        {"name": "+Z", "offset_vec": [0, 0, sampling_offset], "ray_vec": [0, 0, -sampling_offset - obj_aabb_extent[2]]},
        {"name": "-Z", "offset_vec": [0, 0, -sampling_offset], "ray_vec": [0, 0, sampling_offset + obj_aabb_extent[2]]}
    ]

    front_direction = None
    best_non_base_hit_ratio = 0
    front_links_in_each_directions = {}

    for direction in directions:
        log(f"\nTesting direction: {direction['name']}", 2)
        
        # Define grid dimensions based on the perpendicular planes
        if direction["name"] in ["+X", "-X"]:
            n1 = int(obj_aabb_extent[1] / sampling_width)
            n2 = int(obj_aabb_extent[2] / sampling_width)
            axis1, axis2 = 1, 2  # Y and Z axes
        elif direction["name"] in ["+Y", "-Y"]:
            n1 = int(obj_aabb_extent[0] / sampling_width)
            n2 = int(obj_aabb_extent[2] / sampling_width)
            axis1, axis2 = 0, 2  # X and Z axes
        else:  # "+Z", "-Z"
            n1 = int(obj_aabb_extent[0] / sampling_width)
            n2 = int(obj_aabb_extent[1] / sampling_width)
            axis1, axis2 = 0, 1  # X and Y axes

        if n1 * n2 > max_rays:
            scale_factor = np.sqrt(max_rays / (n1 * n2))
            n1 = max(int(n1 * scale_factor), 5)
            n2 = max(int(n2 * scale_factor), 5)
            
        log(f"  Using grid of {n1}x{n2} = {n1*n2} rays", 2)
        
        # Create grid of starting points
        grid1 = np.linspace(obj_aabb_min[axis1], obj_aabb_max[axis1], n1)
        grid2 = np.linspace(obj_aabb_min[axis2], obj_aabb_max[axis2], n2)
        
        # Initialize the start points array as lists of 3D arrays
        starts = []
        
        # Fill in the grid coordinates
        for val1 in grid1:
            for val2 in grid2:
                # Create a starting point (will be filled based on direction)
                start_point = np.zeros(3)
                
                # Set the axes values from the grid
                if axis1 == 0 and axis2 == 1:
                    start_point[0] = val1
                    start_point[1] = val2
                elif axis1 == 0 and axis2 == 2:
                    start_point[0] = val1
                    start_point[2] = val2
                elif axis1 == 1 and axis2 == 2:
                    start_point[1] = val1
                    start_point[2] = val2
                
                # Set the constant axis value based on direction
                if direction["name"] == "+X":
                    start_point[0] = obj_aabb_max[0] + direction["offset_vec"][0]
                elif direction["name"] == "-X":
                    start_point[0] = obj_aabb_min[0] + direction["offset_vec"][0]
                elif direction["name"] == "+Y":
                    start_point[1] = obj_aabb_max[1] + direction["offset_vec"][1]
                elif direction["name"] == "-Y":
                    start_point[1] = obj_aabb_min[1] + direction["offset_vec"][1]
                elif direction["name"] == "+Z":
                    start_point[2] = obj_aabb_max[2] + direction["offset_vec"][2]
                elif direction["name"] == "-Z":
                    start_point[2] = obj_aabb_min[2] + direction["offset_vec"][2]
                
                starts.append(start_point)
        
        # Create end points by adding the ray vector
        ends = [start + np.array(direction["ray_vec"]) for start in starts]
        
        front_links = set()
        # Perform ray test batch
        results = perform_raycast(starts, ends)
        
        # Analyze results
        total_hits = 0
        base_hits = 0
        non_base_hits = 0
        
        for result in results:
            if result["hit"]:
                total_hits += 1
                
                # Get the hit rigid body USD path
                hit_rigid_body_path = result["rigidBody"]
                
                # Extract the exact link name from the rigid body path
                path_parts = hit_rigid_body_path.split('/')
                hit_link_name = path_parts[-1] if len(path_parts) > 0 else None
                
                # Check if the hit is exactly on the base link
                is_base_hit = (hit_link_name == base_link_name)
                
                if is_base_hit:
                    base_hits += 1
                else:
                    non_base_hits += 1
                    front_links.add(hit_link_name)

        # Calculate ratio of non-base hits to total hits
        non_base_hit_ratio = non_base_hits / total_hits if total_hits > 0 else 0
        front_links_in_each_directions[direction["name"]] = front_links
     
        log(f"  Total hits: {total_hits}", 2)
        log(f"  Base hits: {base_hits}", 2)
        log(f"  Non-base hits: {non_base_hits}", 2)
        log(f"  Non-base hit ratio: {non_base_hit_ratio:.4f}", 2)
        
        # Check if this is the best front direction candidate so far
        if non_base_hit_ratio > best_non_base_hit_ratio:
            best_non_base_hit_ratio = non_base_hit_ratio
            front_direction = direction["name"]

    log(f"\nIdentified front direction: {front_direction}")
    log(f"Non-base hit ratio: {best_non_base_hit_ratio:.4f}")
    front_links = front_links_in_each_directions[f"{front_direction}"]
    log(f"Front links in {front_direction}: {front_links}")
    
    return front_direction, front_links

def group_meshes(obj, front_direction, front_links, base_link_name, margin_percentage=0.1, ray_sampling_width=0.00001, max_rays_per_link=500):
    """Step 3: Group meshes by shooting rays from the front direction"""
    log("==================== Step3: Group Meshes ====================")
    # Initialize groups with UnionFind
    uf = UnionFind()
    
    obj_aabb_extent = obj.aabb_extent
    # Determine which axes to use based on the front direction
    if front_direction in ["+X", "-X"]:
        perpendicular_axes = [1, 2]  # Y and Z axes
    elif front_direction in ["+Y", "-Y"]:
        perpendicular_axes = [0, 2]  # X and Z axes
    else:  # "+Z", "-Z"
        perpendicular_axes = [0, 1]  # X and Y axes

    # Get AABBs for each front link
    front_link_aabbs = {}
    for link_name in front_links:
        link_prim = obj.links[link_name]
        front_link_aabbs[link_name] = {
            "min": link_prim.aabb[0],
            "max": link_prim.aabb[1],
            "center": link_prim.aabb_center,
            "extent": link_prim.aabb_extent
        }

    # Direction vector based on front direction
    ray_direction_vector = np.zeros(3)
    ray_offset = 1.0

    if front_direction == "+X":
        ray_direction_vector[0] = -ray_offset - obj_aabb_extent[0] * 1.5
    elif front_direction == "-X":
        ray_direction_vector[0] = ray_offset + obj_aabb_extent[0] * 1.5
    elif front_direction == "+Y":
        ray_direction_vector[1] = -ray_offset - obj_aabb_extent[1] * 1.5
    elif front_direction == "-Y":
        ray_direction_vector[1] = ray_offset + obj_aabb_extent[1] * 1.5
    elif front_direction == "+Z":
        ray_direction_vector[2] = -ray_offset - obj_aabb_extent[2] * 1.5
    elif front_direction == "-Z":
        ray_direction_vector[2] = ray_offset + obj_aabb_extent[2] * 1.5

    # Process each front link
    log(f"Processing {len(front_links)} front links...")

    for link_name in front_links:
        if link_name not in front_link_aabbs:
            continue
            
        aabb = front_link_aabbs[link_name]
        log(f"\nRay testing for link: {link_name}", 2)
        
        # Calculate shrunk boundaries with margin
        min_bounds = np.copy(aabb["min"])
        max_bounds = np.copy(aabb["max"])
        
        for axis in perpendicular_axes:
            axis_range = aabb["max"][axis] - aabb["min"][axis]
            margin = axis_range * margin_percentage
            min_bounds[axis] += margin
            max_bounds[axis] -= margin
        
        # Define grid dimensions for the perpendicular plane
        axis1, axis2 = perpendicular_axes
        n1 = max(int((max_bounds[axis1] - min_bounds[axis1]) / ray_sampling_width), 2)
        n2 = max(int((max_bounds[axis2] - min_bounds[axis2]) / ray_sampling_width), 2)
        
        # Adjust grid size if too many rays
        if n1 * n2 > max_rays_per_link:
            scale_factor = np.sqrt(max_rays_per_link / (n1 * n2))
            n1 = max(int(n1 * scale_factor), 2)
            n2 = max(int(n2 * scale_factor), 2)
        
        log(f"  Using grid of {n1}x{n2} = {n1*n2} rays", 2)
        
        # Create grid of starting points
        grid1 = np.linspace(min_bounds[axis1], max_bounds[axis1], n1)
        grid2 = np.linspace(min_bounds[axis2], max_bounds[axis2], n2)
        
        # Create ray starting points
        starts = []
        
        # Fill in the grid coordinates
        for val1 in grid1:
            for val2 in grid2:
                # Create a starting point
                start_point = np.zeros(3)
                
                # Set grid values
                start_point[axis1] = val1
                start_point[axis2] = val2
                
                # Set position along the ray direction
                if front_direction == "+X":
                    start_point[0] = obj.aabb[1][0] + ray_offset
                elif front_direction == "-X":
                    start_point[0] = obj.aabb[0][0] - ray_offset
                elif front_direction == "+Y":
                    start_point[1] = obj.aabb[1][1] + ray_offset
                elif front_direction == "-Y":
                    start_point[1] = obj.aabb[0][1] - ray_offset
                elif front_direction == "+Z":
                    start_point[2] = obj.aabb[1][2] + ray_offset
                elif front_direction == "-Z":
                    start_point[2] = obj.aabb[0][2] - ray_offset
                
                starts.append(start_point)
        
        # Create end points by adding the ray vector
        ends = [start + ray_direction_vector for start in starts]
        
        # Perform ray test batch
        ray_results = perform_raycast(starts, ends, only_closest=False)
        
        if len(ray_results) > 0 and verbose >= 3:
            log(f"  Debug - Number of rays: {len(ray_results)}", 3)
            log(f"  Debug - First ray hits: {len(ray_results[0])}", 3)
        
        # Process results to establish connections between links
        for ray_hits in ray_results:  # Each element is a list of hits for one ray
            hit_bodies = []
            
            # Process all hits for this ray
            for hit in ray_hits:
                if hit["hit"]:  # Check if this hit was successful
                    # Get the rigid body path
                    hit_path = hit["rigidBody"]
                    
                    # Extract link name from the path
                    path_parts = hit_path.split('/')
                    if len(path_parts) > 0:
                        hit_link = path_parts[-1]
                        
                        # Skip base_link
                        if hit_link != base_link_name:
                            hit_bodies.append(hit_link)
                
                # Connect all pairs of links hit by this ray
                for i in range(len(hit_bodies)):
                    for j in range(i + 1, len(hit_bodies)):
                        uf.union(hit_bodies[i], hit_bodies[j])

    # Extract the final groups from UnionFind
    link_groups = {}
    for link_name in front_links:
        if link_name != "base_link":
            root = uf.find(link_name)
            if root not in link_groups:
                link_groups[root] = set()
            link_groups[root].add(link_name)

    # Clean up groups that are subsets of other groups
    final_groups = []
    for root, group in link_groups.items():
        is_subset = False
        for other_root, other_group in link_groups.items():
            if root != other_root and group.issubset(other_group):
                is_subset = True
                break
        
        if not is_subset:
            final_groups.append(group)

    # Print the final groups
    log("\nFinal Link Groups:")
    for i, group in enumerate(final_groups):
        log(f"Group {i+1}: {group}")

    return final_groups

def identify_handles(obj, group, front_dir, hit_depth_tolerance=0.1, ray_sampling_width=0.0008, handle_ray_count=200):
    """Identify handles in a link group by shooting rays"""
    log(f"\nIdentifying handles for group: {group}", 2)
    
    # Get combined AABB for the group
    min_bounds = np.array([float('inf'), float('inf'), float('inf')])
    max_bounds = np.array([float('-inf'), float('-inf'), float('-inf')])
    
    for link_name in group:
        if link_name in obj.links:
            link_aabb_min, link_aabb_max = obj.links[link_name].aabb
            min_bounds = np.minimum(min_bounds, link_aabb_min)
            max_bounds = np.maximum(max_bounds, link_aabb_max)

    group_center = (min_bounds + max_bounds) / 2
    group_extent = max_bounds - min_bounds

    # Determine which axes to use based on the front direction
    if front_dir in ["+X", "-X"]:
        perpendicular_axes = [1, 2]  # Y and Z axes
        front_axis = 0
    elif front_dir in ["+Y", "-Y"]:
        perpendicular_axes = [0, 2]  # X and Z axes
        front_axis = 1
    else:  # "+Z", "-Z"
        perpendicular_axes = [0, 1]  # X and Y axes
        front_axis = 2
    
    # Define ray parameters
    axis1, axis2 = perpendicular_axes
    ray_offset = 0.1
    
    # Create a grid of ray starting points
    n1 = int(group_extent[axis1] / ray_sampling_width)
    n2 = int(group_extent[axis2] / ray_sampling_width)
    
    # Adjust grid size if too many rays
    if n1 * n2 > handle_ray_count:
        scale_factor = np.sqrt(handle_ray_count / (n1 * n2))
        n1 = max(int(n1 * scale_factor), 5)
        n2 = max(int(n2 * scale_factor), 5)
    
    log(f"  Using grid of {n1}x{n2} = {n1*n2} rays for handle detection", 2)
    
    # Create grid coordinates
    grid1 = np.linspace(min_bounds[axis1], max_bounds[axis1], n1)
    grid2 = np.linspace(min_bounds[axis2], max_bounds[axis2], n2)
    
    # Initialize ray starting points
    starts = []
    ray_direction_vector = np.zeros(3)
    
    # Configure ray direction based on front direction
    if front_dir == "+X":
        ray_direction_vector[0] = -group_extent[0] * 1.5
        ray_start_offset = max_bounds[0] + ray_offset
    elif front_dir == "-X":
        ray_direction_vector[0] = group_extent[0] * 1.5
        ray_start_offset = min_bounds[0] - ray_offset
    elif front_dir == "+Y":
        ray_direction_vector[1] = -group_extent[1] * 1.5
        ray_start_offset = max_bounds[1] + ray_offset
    elif front_dir == "-Y":
        ray_direction_vector[1] = group_extent[1] * 1.5
        ray_start_offset = min_bounds[1] - ray_offset
    elif front_dir == "+Z":
        ray_direction_vector[2] = -group_extent[2] * 1.5
        ray_start_offset = max_bounds[2] + ray_offset
    elif front_dir == "-Z":
        ray_direction_vector[2] = group_extent[2] * 1.5
        ray_start_offset = min_bounds[2] - ray_offset
    
    # Create all ray starting points
    for val1 in grid1:
        for val2 in grid2:
            start_point = np.zeros(3)
            start_point[axis1] = val1
            start_point[axis2] = val2
            start_point[front_axis] = ray_start_offset
            starts.append(start_point)
    
    # Create ray endpoints
    ends = [start + ray_direction_vector for start in starts]
    
    # Perform ray test batch
    results = perform_raycast(starts, ends)
    
    sorted_hits = sorted([result for result in results if result["hit"]], key=lambda x: x["distance"])
    if not sorted_hits:
        log("  No hits detected for this group", 2)
        return {"handle": None, "group": group}
        
    min_dist = sorted_hits[0]["distance"]
    pruned_positions = []
    for hit in sorted_hits:
        if hit["distance"] > (min_dist + hit_depth_tolerance):
            break
        pruned_positions.append(hit["position"])

    # Get the mean position -- this will be the tip of the grasping point
    pruned_positions = np.array(pruned_positions)
    grasp_pos_canonical_rotated = pruned_positions.mean(axis=0)
    # Extract handle link name from the rigid body path
    hit_path = sorted_hits[0]["rigidBody"]
    path_parts = hit_path.split('/')
    handle_link = path_parts[-1] if len(path_parts) > 0 else None
    log(f"  Identified handle link: {handle_link}", 2)
    log(f"  Handle center: {grasp_pos_canonical_rotated}", 2)
    
    return {
        "handle": handle_link,
        "group": group,
        "handle_center": grasp_pos_canonical_rotated,
    }

def detect_handles(obj, final_groups, front_direction, hit_depth_tolerance=0.1):
    """Step 4: Detect handles for each group"""
    log("==================== Step4: Detect Handle ====================")
    
    group_handle_results = []
    for group_idx, group in enumerate(final_groups):
        log(f"\nProcessing group {group_idx + 1}/{len(final_groups)}")
        handle_info = identify_handles(
            obj=obj,
            group=group,
            front_dir=front_direction,
            hit_depth_tolerance=hit_depth_tolerance,
            handle_ray_count=500,
        )
        group_handle_results.append(handle_info)

    log("Identify handle results:")
    for idx, result in enumerate(group_handle_results):
        log(f"Group {idx+1}: Handle = {result['handle']}")
        
    return group_handle_results

def determine_joint_type(obj, group_info, front_direction, base_link_name, edge_threshold=0.2):
    """Determine joint type and parameters based on handle position"""
    group = group_info["group"]
    handle = group_info["handle"]
    if handle is None:
        log(f"  No handle detected for group, skipping joint detection", 2)
        group_info["joint_type"] = None
        return group_info
    
    # Get combined AABB for the group
    min_bounds = np.array([float('inf'), float('inf'), float('inf')])
    max_bounds = np.array([float('-inf'), float('-inf'), float('-inf')])
    
    for link_name in group:
        if link_name in obj.links:
            link_aabb_min, link_aabb_max = obj.links[link_name].aabb
            min_bounds = np.minimum(min_bounds, link_aabb_min)
            max_bounds = np.maximum(max_bounds, link_aabb_max)
    
    min_bounds = np.array(min_bounds)
    max_bounds = np.array(max_bounds)
    group_center = (min_bounds + max_bounds) / 2
    group_extent = max_bounds - min_bounds
    
    base_link_aabb_min, base_link_aabb_max = obj.links[base_link_name].aabb
    base_link_aabb_min = np.array(base_link_aabb_min)
    base_link_aabb_max = np.array(base_link_aabb_max)
    base_link_aabb_extent = np.array(obj.links[base_link_name].aabb_extent)
    
    # Handle position
    handle_pos = group_info["handle_center"]
    # Normalize handle position relative to AABB bounds (0 to 1 scale)
    normalized_pos = (handle_pos - min_bounds) / group_extent
    
    # Determine axes based on front direction
    if front_direction in ["+X", "-X"]:
        front_axis = 0
        horizontal_axis = 2  # Z axis
        vertical_axis = 1    # Y axis
    elif front_direction in ["+Y", "-Y"]:
        front_axis = 1
        horizontal_axis = 0  # X axis
        vertical_axis = 2    # Z axis
    else:  # "+Z", "-Z"
        front_axis = 2
        horizontal_axis = 0  # X axis
        vertical_axis = 1    # Y axis
    
    # Get normalized positions for relevant axes
    h_pos = normalized_pos[horizontal_axis]  # Horizontal position (0-1)
    v_pos = normalized_pos[vertical_axis]    # Vertical position (0-1)
    log(f"  h_pos = {h_pos}", 2)
    log(f"  v_pos = {v_pos}", 2)
    
    # Determine joint type based on handle position
    joint_info = {
        "parent": base_link_name,
        "child": handle  # First link in group as representative
    }
    
    # Default RPY (no rotation)
    joint_info["rpy"] = [0.0, 0.0, 0.0]
    revolute_joint_limit = 3.14 * 2.0 / 3.0
    # Check if handle is near edges
    if h_pos < edge_threshold:
        # Left edge - revolute joint along right edge
        joint_info["type"] = "revolute"
        
        # Joint origin (midpoint of right edge)
        origin = np.copy(group_center)
        origin[horizontal_axis] = max_bounds[horizontal_axis]  # Right edge
        
        # Set origin at the front surface based on front direction
        if front_direction.startswith('-'):
            origin[front_axis] = min(base_link_aabb_min[front_axis], base_link_aabb_max[front_axis])
        else:
            origin[front_axis] = max(base_link_aabb_min[front_axis], base_link_aabb_max[front_axis])
        
        # Joint axis (vertical)
        axis = np.zeros(3)
        axis[vertical_axis] = 1  # Rotate around vertical axis
        
        # Adjust axis direction based on front direction
        # This ensures door opens away from front direction
        if front_direction in ["+X", "+Y", "-Z"]:
            joint_info["axis"] = -axis  # Negative direction
            joint_info["range"] = [0.0, revolute_joint_limit]  # 0° to 90°
        elif front_direction in ["-X", "-Y", "+Z"]:
            joint_info["axis"] = axis  # Positive direction
            joint_info["range"] = [0.0, revolute_joint_limit]  # 0° to 90°  
        
            
        joint_info["origin"] = origin
        log(f"  Left edge handle detected - revolute joint along right edge", 2)
        
    elif h_pos > (1 - edge_threshold):
        # Right edge - revolute joint along left edge
        joint_info["type"] = "revolute"
        
        # Joint origin (midpoint of left edge)
        origin = np.copy(group_center)
        origin[horizontal_axis] = min_bounds[horizontal_axis]  # Left edge
        
        # Set origin at the front surface based on front direction
        if front_direction.startswith('-'):
            origin[front_axis] = min(base_link_aabb_min[front_axis], base_link_aabb_max[front_axis])
        else:
            origin[front_axis] = max(base_link_aabb_min[front_axis], base_link_aabb_max[front_axis])
        
        # Joint axis (vertical)
        axis = np.zeros(3)
        axis[vertical_axis] = 1  # Rotate around vertical axis
        
        # Adjust axis direction based on front direction
        # This ensures door opens away from front direction
        if front_direction in ["+X", "+Y", "-Z"]:
            joint_info["axis"] = axis  # Positive direction
            joint_info["range"] = [0.0, revolute_joint_limit]  # 0° to 90°
        elif front_direction in ["-X", "-Y", "+Z"]:
            joint_info["axis"] = -axis  # Negative direction
            joint_info["range"] = [0.0, revolute_joint_limit]  # 0° to 90°
            
        joint_info["origin"] = origin
        log(f"  Right edge handle detected - revolute joint along left edge", 2)
        
    elif v_pos < edge_threshold:
        # Bottom edge - revolute joint along top edge
        joint_info["type"] = "revolute"
        
        # Joint origin (midpoint of top edge)
        origin = np.copy(group_center)
        origin[vertical_axis] = max_bounds[vertical_axis]  # Top edge
        
        # Set origin at the front surface based on front direction
        if front_direction.startswith('-'):
            origin[front_axis] = min(base_link_aabb_min[front_axis], base_link_aabb_max[front_axis])
        else:
            origin[front_axis] = max(base_link_aabb_min[front_axis], base_link_aabb_max[front_axis])
        
        # Joint axis (horizontal)
        axis = np.zeros(3)
        axis[horizontal_axis] = 1  # Rotate around horizontal axis
        
        # Adjust axis direction based on front direction
        # This ensures door opens away from front direction
        if front_direction in ["+X", "+Y", "-Z"]:
            joint_info["axis"] = axis  # Positive direction
            joint_info["range"] = [0.0, revolute_joint_limit]  # 0° to 90°
        elif front_direction in ["-X", "-Y", "+Z"]:
            joint_info["axis"] = -axis  # Negative direction
            joint_info["range"] = [0.0, revolute_joint_limit]  # 0° to 90°
            
        joint_info["origin"] = origin
        log(f"  Bottom edge handle detected - revolute joint along top edge", 2)
        
    elif v_pos > (1 - edge_threshold):
        # Top edge - revolute joint along bottom edge
        joint_info["type"] = "revolute"
        
        # Joint origin (midpoint of bottom edge)
        origin = np.copy(group_center)
        origin[vertical_axis] = min_bounds[vertical_axis]  # Bottom edge
        
        # Set origin at the front surface based on front direction
        if front_direction.startswith('-'):
            origin[front_axis] = min(base_link_aabb_min[front_axis], base_link_aabb_max[front_axis])
        else:
            origin[front_axis] = max(base_link_aabb_min[front_axis], base_link_aabb_max[front_axis])
        
        # Joint axis (horizontal)
        axis = np.zeros(3)
        axis[horizontal_axis] = 1  # Rotate around horizontal axis
        
        # Adjust axis direction based on front direction
        # This ensures door opens away from front direction
        if front_direction in ["+X", "+Y", "-Z"]:
            joint_info["axis"] = -axis  # Negative direction
            joint_info["range"] = [0.0, revolute_joint_limit]  # 0° to 90°
        elif front_direction in ["-X", "-Y", "+Z"]:
            joint_info["axis"] = axis  # Positive direction
            joint_info["range"] = [0.0, revolute_joint_limit]  # 0° to 90° 
            
        joint_info["origin"] = origin
        log(f"  Top edge handle detected - revolute joint along bottom edge", 2)
        
    else:
        # Center area - prismatic joint along front direction
        joint_info["type"] = "prismatic"
        
        # Joint origin (center of group)
        origin = np.copy(group_center)
        
        # Set origin at the front surface based on front direction
        if front_direction.startswith('-'):
            origin[front_axis] = min(base_link_aabb_min[front_axis], base_link_aabb_max[front_axis])
        else:
            origin[front_axis] = max(base_link_aabb_min[front_axis], base_link_aabb_max[front_axis])
            
        joint_info["origin"] = origin
        
        # Joint axis (front direction)
        axis = np.zeros(3)
        axis_index = front_axis
        
        if front_direction.startswith('+'):
            axis_direction = 1
            joint_info["range"] = [0.0, base_link_aabb_extent[front_axis]]
        else:
            axis_direction = -1
            joint_info["range"] = [0.0, base_link_aabb_extent[front_axis]]
        
        axis[axis_index] = axis_direction
        joint_info["axis"] = axis
        log(f"  Center handle detected - prismatic joint along front direction", 2)
    
    # Add joint information to the group info
    group_info["joint"] = joint_info
    
    return group_info

def determine_joint_types(obj, group_handle_results, front_direction, base_link_name, edge_threshold=0.30):
    """Step 5: Determine joint types for each group"""
    log("\n==================== Step5: Determine Joint Types ====================")
    
    for i, group_info in enumerate(group_handle_results):
        log(f"\nDetermining joint for group {i+1}:")
        group_handle_results[i] = determine_joint_type(
            obj=obj,
            group_info=group_info, 
            front_direction=front_direction,
            base_link_name=base_link_name,
            edge_threshold=edge_threshold,
        )

    # Print final results with joint information
    log("\nFinal Results with Joint Information:")
    for idx, result in enumerate(group_handle_results):
        log(f"Group {idx + 1}:")
        log(f"  Links: {result['group']}")
        if result['handle']:
            log(f"  Handle link: {result['handle']}")
            log(f"  Handle center: {result['handle_center']}")
            
            if 'joint' in result and result['joint'] is not None:
                joint = result['joint']
                log(f"  Joint type: {joint['type']}")
                log(f"  Joint origin: {joint['origin']}")
                log(f"  Joint axis: {joint['axis']}")
                log(f"  Joint rpy: {joint['rpy']}")
                log(f"  Joint range: {joint['range']}")
                log(f"  Parent link: {joint['parent']}")
        else:
            log("  No handle detected, no joint assigned")
        log("")

    # Final results
    urdf_analysis_results = {
        "front_direction": front_direction,
        "base_link": base_link_name,
        "groups": group_handle_results
    }
    
    return urdf_analysis_results

def create_rotation_matrix(front_direction):
    """Create rotation matrix based on front direction"""
    # Create rotation matrix based on front direction
    rotation_matrix = np.eye(3)

    # Set rotation based on front direction
    if front_direction == "+X":
        pass  # No rotation needed
    elif front_direction == "-X":
        # 180 degrees around Z
        rotation_matrix = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
    elif front_direction == "+Y":
        # -90 degrees around Z
        rotation_matrix = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ])
    elif front_direction == "-Y":
        # 90 degrees around Z
        rotation_matrix = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
    elif front_direction == "+Z":
        # -90 degrees around Y
        rotation_matrix = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])
    elif front_direction == "-Z":
        # 90 degrees around Y
        rotation_matrix = np.array([
            [0, 0, -1],
            [0, 1, 0],
            [1, 0, 0]
        ])
        
    return rotation_matrix

def rotate_and_center_urdf(model_name, base_link_name, front_direction):
    """Step 6: Rotate and center the URDF"""
    log("\n==================== Step6: Rotate and Center URDF ====================")
    
    rotation_matrix = create_rotation_matrix(front_direction)

    # First, make a copy of the original URDF file
    log(f"Creating a copy of the original URDF file: {checkpoint_path_1}")
    shutil.copy2(original_urdf_path, checkpoint_path_1)

    # Parse the URDF file
    tree = ET.parse(checkpoint_path_1)
    root = tree.getroot()

    # Find the joint where parent is "base_link" and child is our actual base link
    base_joint = None
    for joint in root.findall('joint'):
        parent = joint.find('parent')
        child = joint.find('child')
        if (parent is not None and parent.attrib['link'] == "base_link" and
            child is not None and child.attrib['link'] == base_link_name):
            base_joint = joint
            break

    if base_joint is None:
        log(f"Error: No joint found connecting 'base_link' to '{base_link_name}'")
        return None, None

    # Get the base joint origin
    base_origin = base_joint.find('origin')
    if base_origin is None:
        log(f"Error: No origin found for base link joint")
        return None, None

    # Get translation (position) from the base joint
    if 'xyz' in base_origin.attrib:
        translation = np.array([float(x) for x in base_origin.attrib['xyz'].split()])
    else:
        translation = np.zeros(3)

    # Process all joints in the URDF
    for joint in root.findall('joint'):
        origin = joint.find('origin')
        if origin is None:
            continue
        
        # Process position (xyz)
        if 'xyz' in origin.attrib:
            position = np.array([float(x) for x in origin.attrib['xyz'].split()])
            
            # Apply transformation: first rotation, then translation
            new_position = np.dot(rotation_matrix, position - translation)
            
            # Update position
            origin.attrib['xyz'] = f"{new_position[0]:.6f} {new_position[1]:.6f} {new_position[2]:.6f}"
        
        # Process orientation (rpy)
        if 'rpy' in origin.attrib:
            # Extract roll, pitch, yaw
            rpy = np.array([float(r) for r in origin.attrib['rpy'].split()])
            
            # Convert to rotation matrix
            cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
            cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
            cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
            
            # Create rotation matrices for roll, pitch, yaw
            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            
            # Combined rotation matrix
            joint_rotation = np.dot(Rz, np.dot(Ry, Rx))
            
            # Apply our rotation
            new_rotation = np.dot(rotation_matrix, joint_rotation)
            
            # Convert back to roll, pitch, yaw
            # This is a simplified conversion that works in most cases
            sy = np.sqrt(new_rotation[0,0] * new_rotation[0,0] + new_rotation[1,0] * new_rotation[1,0])
            
            if sy > 1e-6:
                roll = np.arctan2(new_rotation[2,1], new_rotation[2,2])
                pitch = np.arctan2(-new_rotation[2,0], sy)
                yaw = np.arctan2(new_rotation[1,0], new_rotation[0,0])
            else:
                # Gimbal lock case
                roll = np.arctan2(-new_rotation[1,2], new_rotation[1,1])
                pitch = np.arctan2(-new_rotation[2,0], sy)
                yaw = 0
            
            # Update orientation
            origin.attrib['rpy'] = f"{roll:.6f} {pitch:.6f} {yaw:.6f}"
            
        # # Process joint axis if present
        # axis = joint.find('axis')
        # if axis is not None and 'xyz' in axis.attrib:
        #     axis_vec = np.array([float(x) for x in axis.attrib['xyz'].split()])
            
        #     # Apply rotation to axis
        #     new_axis = np.dot(rotation_matrix, axis_vec)
            
        #     # Normalize
        #     new_axis = new_axis / np.linalg.norm(new_axis)
            
        #     # Update axis
        #     axis.attrib['xyz'] = f"{new_axis[0]:.6f} {new_axis[1]:.6f} {new_axis[2]:.6f}"

    # Write the transformed URDF
    write_urdf_to_file(root, checkpoint_path_1)
    log(f"Centered URDF saved to {checkpoint_path_1}")
    
    return rotation_matrix, translation

def write_urdf_to_file(root, file_path):
    """Helper function to write URDF to file with pretty formatting"""
    rough_string = ET.tostring(root, 'utf-8')
    parsed = minidom.parseString(rough_string)
    pretty_xml = parsed.toprettyxml(indent="  ")

    # Clean up the XML output
    lines = pretty_xml.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    clean_xml = '\n'.join(non_empty_lines)

    with open(file_path, 'w') as f:
        f.write(clean_xml)

def add_articulated_joints(model_name, urdf_analysis_results, rotation_matrix, translation):
    """Step 7: Add articulated joints to the URDF"""
    log("\n==================== Step7: Rotate and Center articulate joints ====================")
    
    # Parse the URDF file
    tree = ET.parse(checkpoint_path_1)
    root = tree.getroot()
    
    # Move articulated joints for each group to the right global position
    for i, group_info in enumerate(urdf_analysis_results["groups"]):
        if 'joint' not in group_info or group_info['joint'] is None:
            continue
            
        joint_info = group_info['joint']
        joint_type = joint_info['type']
        parent_link = "base_link"
        child_link = joint_info['child'] # set handle as child
        joint_origin = np.array(joint_info['origin'])
        joint_axis = np.array(joint_info['axis'])
        joint_rpy = joint_info['rpy']
        joint_low_limit = joint_info['range'][0]
        joint_high_limit = joint_info['range'][1]
        
        log(f"Adding {joint_type} joint: {parent_link} → {child_link}")
        
        # Transform joint origin and axis
        transformed_origin = np.dot(rotation_matrix, joint_origin - translation)
        # transformed_axis = np.dot(rotation_matrix, joint_axis)
        # transformed_axis = transformed_axis / np.linalg.norm(transformed_axis)
        
        # Convert to rotation matrix
        cr, sr = np.cos(joint_rpy[0]), np.sin(joint_rpy[0])
        cp, sp = np.cos(joint_rpy[1]), np.sin(joint_rpy[1])
        cy, sy = np.cos(joint_rpy[2]), np.sin(joint_rpy[2])
        
        # Create rotation matrices for roll, pitch, yaw
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        
        # Combined rotation matrix
        joint_rotation = np.dot(Rz, np.dot(Ry, Rx))
        
        # Apply our rotation
        new_rotation = np.dot(rotation_matrix, joint_rotation)
        
        # Convert back to roll, pitch, yaw
        sy = np.sqrt(new_rotation[0,0] * new_rotation[0,0] + new_rotation[1,0] * new_rotation[1,0])
        
        if sy > 1e-6:
            roll = np.arctan2(new_rotation[2,1], new_rotation[2,2])
            pitch = np.arctan2(-new_rotation[2,0], sy)
            yaw = np.arctan2(new_rotation[1,0], new_rotation[0,0])
        else:
            # Gimbal lock case
            roll = np.arctan2(-new_rotation[1,2], new_rotation[1,1])
            pitch = np.arctan2(-new_rotation[2,0], sy)
            yaw = 0
        
        # Create a new joint element
        new_joint = ET.SubElement(root, 'joint')
        new_joint.attrib['name'] = f"articulated_group{i+1}_joint"
        new_joint.attrib['type'] = joint_type
        
        # Add parent and child elements
        parent_elem = ET.SubElement(new_joint, 'parent')
        parent_elem.attrib['link'] = parent_link
        
        child_elem = ET.SubElement(new_joint, 'child')
        child_elem.attrib['link'] = child_link
        
        # Add origin element
        origin_elem = ET.SubElement(new_joint, 'origin')
        origin_elem.attrib['xyz'] = f"{transformed_origin[0]:.6f} {transformed_origin[1]:.6f} {transformed_origin[2]:.6f}"
        origin_elem.attrib['rpy'] = f"{roll:.6f} {pitch:.6f} {yaw:.6f}"
        
        # Add axis element
        axis_elem = ET.SubElement(new_joint, 'axis')
        axis_elem.attrib['xyz'] = f"{joint_axis[0]:.6f} {joint_axis[1]:.6f} {joint_axis[2]:.6f}"
        
        # Add limit element
        limit_elem = ET.SubElement(new_joint, 'limit')
        if joint_type == 'revolute':
            limit_elem.attrib['lower'] = f"{joint_low_limit}"
            limit_elem.attrib['upper'] = f"{joint_high_limit}"
            limit_elem.attrib['effort'] = "100.0"
            limit_elem.attrib['velocity'] = "1.0"
        else:  # prismatic
            limit_elem.attrib['lower'] = f"{joint_low_limit}"
            limit_elem.attrib['upper'] = f"{joint_high_limit}"
            limit_elem.attrib['effort'] = "100.0"
            limit_elem.attrib['velocity'] = "0.5"
            
    # Write the updated URDF
    write_urdf_to_file(root, checkpoint_path_1)
    

def apply_transform(pos, rpy, rel_pos, rel_rpy):
    """Apply a relative transform to a position and orientation"""
    # Convert RPY to quaternion
    quat = T.euler2quat(rpy)
    rel_quat = T.euler2quat(rel_rpy)
    
    # Apply rotation to position
    rot_matrix = T.quat2mat(rel_quat)
    rotated_pos = rot_matrix.dot(pos)
    
    # Add the relative position
    new_pos = rotated_pos + rel_pos
    
    # Combine quaternions for rotation
    new_quat = T.quat_multiply(rel_quat, quat)
    
    # Convert back to RPY
    new_rpy = T.quat2euler(new_quat)
    
    return new_pos, new_rpy

def transfer_links_to_joint_coordinate(model_name, urdf_analysis_results):
    """Step 8: Transfer links to joint coordinate system and group them"""
    log("\n==================== Step8: Transfer links to joint coordinate and group it ====================")
    
    shutil.copy2(checkpoint_path_1, checkpoint_path_2)
    log(f"Making a copy of {checkpoint_path_1} to {checkpoint_path_2}")
    
    # Parse the URDF file
    tree = ET.parse(checkpoint_path_2)
    root = tree.getroot()
    
    # Create a mapping of link names to their XML elements
    link_map = {link.attrib['name']: link for link in root.findall('link')}

    # Find all joints with base_link as parent
    fixed_joints_to_remove = []

    # Process each group
    for group_info in urdf_analysis_results["groups"]:
        handle = group_info['handle']
        if not handle or 'joint' not in group_info or group_info['joint'] is None:
            continue
            
        group_links = group_info['group']
        
        log(f"Processing group with handle: {handle}")
        
        # Find the non-fixed joint that connects to the handle
        connecting_joint = None
        for joint in root.findall('joint'):
            if (joint.attrib.get('type', '') != 'fixed' and 
                joint.find('child').attrib['link'] == handle and
                joint.find('parent').attrib['link'] == "base_link"):
                connecting_joint = joint
                break
        
        if connecting_joint is None:
            log(f"  Warning: Could not find non-fixed joint connecting to handle {handle}", 2)
            continue
        
        # Get joint position and orientation
        joint_origin = connecting_joint.find('origin')
        joint_pos = np.zeros(3)
        joint_rpy = np.zeros(3)
        
        if 'xyz' in joint_origin.attrib:
            joint_pos = np.array([float(x) for x in joint_origin.attrib['xyz'].split()])
        
        if 'rpy' in joint_origin.attrib:
            joint_rpy = np.array([float(r) for r in joint_origin.attrib['rpy'].split()])
        
        # Convert joint RPY to quaternion
        joint_quat = T.euler2quat(joint_rpy)
        
        # Create a new merged link name
        merged_link_name = f"{handle}_merged"
        
        # Create the new merged link element
        merged_link = ET.SubElement(root, 'link', {'name': merged_link_name})
        
        # Find all fixed joints connecting base_link to links in this group
        group_fixed_joints = []
        for joint in root.findall('joint'):
            if (joint.attrib.get('type', '') == 'fixed' and 
                joint.find('parent').attrib['link'] == "base_link" and
                joint.find('child').attrib['link'] in group_links):
                group_fixed_joints.append(joint)
        
        # Process and transform each link in the group
        for link_name in group_links:
            if link_name not in link_map:
                log(f"  Warning: Link {link_name} not found in URDF", 2)
                continue
            
            link = link_map[link_name]
            
            # Find the fixed joint connecting this link to base_link
            fixed_joint = None
            for joint in group_fixed_joints:
                if joint.find('child').attrib['link'] == link_name:
                    fixed_joint = joint
                    break
                
            # Get fixed joint position and orientation
            fixed_origin = fixed_joint.find('origin')
            fixed_pos = np.zeros(3)
            fixed_rpy = np.zeros(3)
            
            if fixed_origin is not None:
                if 'xyz' in fixed_origin.attrib:
                    fixed_pos = np.array([float(x) for x in fixed_origin.attrib['xyz'].split()])
                
                if 'rpy' in fixed_origin.attrib:
                    fixed_rpy = np.array([float(r) for r in fixed_origin.attrib['rpy'].split()])
            
            # Convert fixed joint RPY to quaternion
            fixed_quat = T.euler2quat(fixed_rpy)
            fixed_joints_to_remove.append(fixed_joint)
            
            # Calculate relative transform from joint to link
            rel_pos, rel_quat = T.relative_pose_transform(
                fixed_pos, fixed_quat, joint_pos, joint_quat)
            
            # Convert the quaternion back to RPY for URDF
            rel_rpy = T.quat2euler(rel_quat)
            
            if verbose >= 2:
                log(f"  Link: {link_name}", 2)
                log(f"  Non-fixed joint XYZ: {joint_pos}, RPY: {joint_rpy}", 3)
                log(f"  Fixed joint XYZ: {fixed_pos}, RPY: {fixed_rpy}", 3)
                log(f"  Relative XYZ: {rel_pos}, RPY: {rel_rpy}", 3)
            
            # Copy and transform visual elements
            for visual in link.findall('visual'):
                # Deep copy the visual element
                new_visual = copy.deepcopy(visual)
                
                # Get or create visual origin
                visual_origin = new_visual.find('origin')
                if visual_origin is None:
                    visual_origin = ET.SubElement(new_visual, 'origin')
                    visual_origin.attrib['xyz'] = "0 0 0"
                    visual_origin.attrib['rpy'] = "0 0 0"
                
                # Get current visual origin values
                vis_xyz = np.zeros(3)
                vis_rpy = np.zeros(3)
                
                if 'xyz' in visual_origin.attrib:
                    vis_xyz = np.array([float(x) for x in visual_origin.attrib['xyz'].split()])
                
                if 'rpy' in visual_origin.attrib:
                    vis_rpy = np.array([float(r) for r in visual_origin.attrib['rpy'].split()])
                
                # Apply the relative transform to the visual origin
                new_xyz, new_rpy = apply_transform(vis_xyz, vis_rpy, rel_pos, rel_rpy)
                
                # Update the visual origin
                visual_origin.attrib['xyz'] = f"{new_xyz[0]:.6f} {new_xyz[1]:.6f} {new_xyz[2]:.6f}"
                visual_origin.attrib['rpy'] = f"{new_rpy[0]:.6f} {new_rpy[1]:.6f} {new_rpy[2]:.6f}"
                
                # Add the transformed visual to the merged link
                merged_link.append(new_visual)
            
            # Copy and transform collision elements
            for collision in link.findall('collision'):
                # Deep copy the collision element
                new_collision = copy.deepcopy(collision)
                
                # Get or create collision origin
                collision_origin = new_collision.find('origin')
                if collision_origin is None:
                    collision_origin = ET.SubElement(new_collision, 'origin')
                    collision_origin.attrib['xyz'] = "0 0 0"
                    collision_origin.attrib['rpy'] = "0 0 0"
                
                # Get current collision origin values
                col_xyz = np.zeros(3)
                col_rpy = np.zeros(3)
                
                if 'xyz' in collision_origin.attrib:
                    col_xyz = np.array([float(x) for x in collision_origin.attrib['xyz'].split()])
                
                if 'rpy' in collision_origin.attrib:
                    col_rpy = np.array([float(r) for r in collision_origin.attrib['rpy'].split()])
                
                # Apply the relative transform to the collision origin
                new_xyz, new_rpy = apply_transform(col_xyz, col_rpy, rel_pos, rel_rpy)
                
                # Update the collision origin
                collision_origin.attrib['xyz'] = f"{new_xyz[0]:.6f} {new_xyz[1]:.6f} {new_xyz[2]:.6f}"
                collision_origin.attrib['rpy'] = f"{new_rpy[0]:.6f} {new_rpy[1]:.6f} {new_rpy[2]:.6f}"
                
                # Add the transformed collision to the merged link
                merged_link.append(new_collision)
        
        # Update the non-fixed joint's child link to the merged link
        connecting_joint.find('child').attrib['link'] = merged_link_name
        
        # Schedule the original links in the group for removal
        for link_name in group_links:
            if link_name in link_map:
                root.remove(link_map[link_name])

    # Remove all fixed joints that were connecting to base_link
    for joint in fixed_joints_to_remove:
        if joint in root:
            root.remove(joint)
            
    # Write the updated URDF to the new file
    write_urdf_to_file(root, checkpoint_path_2)

def urdf_articulation_pipeline(category, model_name, print_output=True):
    """
    Main function that runs the complete URDF articulation pipeline
    
    Args:
        model_name: Name of the model to process
        print_output: Whether to print verbose output
    
    Returns:
        urdf_analysis_results: Dictionary with analysis results
    """
    global verbose
    verbose = 2 if print_output else 0
    
    # Set OmniGibson to headless mode
    gm.HEADLESS = True

    dataset_root = gm.CUSTOM_DATASET_PATH
    global original_urdf_path, checkpoint_path_1, checkpoint_path_2
    original_urdf_path = os.path.join(dataset_root, "objects", category, model_name, "urdf", f"{model_name}.urdf") 
    checkpoint_path_1 = os.path.join(dataset_root, "objects", category, model_name, "urdf", f"{model_name}_c1.urdf") 
    checkpoint_path_2 = os.path.join(dataset_root, "objects", category, model_name, "urdf", f"{model_name}_c2.urdf") 

    # Initialize OmniGibson
    og.launch()
    scene = Scene(use_floor_plane=False)
    og.sim.import_scene(scene)
    
    # Load the object
    obj = DatasetObject(
        name="objvXL",
        category=category,
        model=model_name,
        dataset_type=DatasetType.CUSTOM,
        fixed_base=True,
    )
    scene.add_object(obj)
    og.sim.play()
    
    try:
        # Execute the pipeline steps
        # Step 1: Determine base mesh
        base_link_name = determine_base_mesh(obj)
        
        # Step 2: Specify front direction
        front_direction, front_links = specify_front_direction(obj, base_link_name)
        
        # Step 3: Group meshes
        final_groups = group_meshes(obj, front_direction, front_links, base_link_name)
        
        # Step 4: Detect handles
        group_handle_results = detect_handles(obj, final_groups, front_direction)
        
        # Step 5: Determine joint types
        urdf_analysis_results = determine_joint_types(obj, group_handle_results, front_direction, base_link_name)
        
        # Step 6: Rotate and center URDF
        rotation_matrix, translation = rotate_and_center_urdf(model_name, base_link_name, front_direction)

        # Step 7: Add articulated joints
        add_articulated_joints(model_name, urdf_analysis_results, rotation_matrix, translation)
        
        # Step 8: Transfer links to joint coordinate system
        transfer_links_to_joint_coordinate(model_name, urdf_analysis_results)
        
        log("==================== Successfully Converted! ====================")
        
        return urdf_analysis_results
        
    except Exception as e:
        log(f"Error in URDF articulation pipeline: {e}")
        return None
    finally:
        # Clean up OmniGibson
        og.shutdown()

if __name__ == "__main__":
    # Example usage
    CATEGORY = "objaverse"
    MODEL_NAME = "AAAAAA"
    urdf_analysis_results = urdf_articulation_pipeline(CATEGORY, MODEL_NAME, print_output=True)

    # AAAAAA - rusty_fridge.glb
    # AAAAAB - tall_closet.glb
    # AAAAAC - tall_closet.glb
    # AAAAAD - paper_drawer.glb
    # AAAAAE - ikea_drawer.glb
    # AAAAAF - paper_drawer.glb