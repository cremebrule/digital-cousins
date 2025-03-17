import trimesh
import numpy as np
import pathlib
from pathlib import Path
import torch
import click
import os
import glob
import cv2
import json
import shutil
from digital_cousins.models.feature_matcher import FeatureMatcher
from groundingdino.util.inference import load_image, annotate

# Import PyVista only when needed
pyvista = None

def init_pyvista():
    """Initialize PyVista with proper settings for headless rendering"""
    global pyvista
    
    if pyvista is None:
        # Now import PyVista
        import pyvista as pv
        pyvista = pv
        
        # Configure PyVista
        pyvista.OFF_SCREEN = True
        
        try:
            # Try to start virtual framebuffer
            pyvista.start_xvfb(wait=1)
        except Exception as e:
            print(f"Warning: Could not start virtual framebuffer: {e}")
            # Continue anyway as we're using software rendering
            
def rescale_and_center_mesh(mesh, target_size=1.0, center=True):
    """
    Process a mesh to make it suitable for visualization.
    
    Args:
        mesh: Input mesh
        target_size: Desired maximum dimension size in meters
        center: Whether to center the mesh at origin
        
    Returns:
        Processed mesh
    """
    processed_mesh = mesh.copy()
    
    # Calculate and apply scale
    bbox_size = processed_mesh.bounding_box.extents
    current_max_size = max(bbox_size)
    
    # Calculate scale factor to achieve target size
    scale_factor = target_size / current_max_size
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= scale_factor
    processed_mesh.apply_transform(scale_matrix)
    
    # Center the mesh if requested
    if center:
        # Use center of mass if available, otherwise use centroid
        if processed_mesh.is_watertight:
            center_point = processed_mesh.center_mass
        else:
            center_point = processed_mesh.centroid
            
        translation = -center_point
        processed_mesh.apply_translation(translation)
    
    return processed_mesh

def process_mesh(asset_path):
    """
    Process a 3D file, scale it, and save it for visualization.
    
    Args:
        asset_path: Path to the input 3D file (.obj, .dae, .glb, .gltf)
        output_dir: Directory to save the processed mesh (default: temporary directory)
        
    Returns:
        Path to the processed mesh file
    """
    # Create images directory at the same level as the obj files
    asset_path = pathlib.Path(asset_path)
    output_dir = asset_path.parent
    
    # Load the 3D file
    scene = trimesh.load(asset_path)

    original_filename = os.path.basename(asset_path)
    base_name = os.path.splitext(original_filename)[0]
    output_path = os.path.join(output_dir, f"{base_name}.obj")

    if isinstance(scene, trimesh.Trimesh):
        # Single mesh case
        processed_mesh = rescale_and_center_mesh(scene)
        processed_mesh.export(output_path, file_type='obj')
        return output_path
    
    elif isinstance(scene, trimesh.Scene):
        # Handle multiple meshes in a scene
        meshes = []
        
        # Extract all meshes from the scene with their transforms
        if isinstance(scene.geometry, dict):
            for name, geometry in scene.geometry.items():
                # Case 1: Direct trimesh object
                if isinstance(geometry, trimesh.Trimesh):
                    # Get transform for this geometry
                    transform = np.eye(4)
                    for node_name in scene.graph.nodes_geometry:
                        if scene.graph[node_name][1] == name:
                            transform = scene.graph.get(node_name)[0]
                            break
                    
                    # Apply transform and add to meshes list
                    mesh_copy = geometry.copy()
                    mesh_copy.apply_transform(transform)
                    meshes.append(mesh_copy)
                
                # Case 2: List or tuple of meshes
                elif isinstance(geometry, (list, tuple)):
                    for sub_idx, submesh in enumerate(geometry):
                        if isinstance(submesh, trimesh.Trimesh):
                            # For submeshes, we'll use identity transform if not found
                            meshes.append(submesh.copy())
        
        # Simpler alternative approach: just use scene.dump() if available
        if not meshes and hasattr(scene, 'dump'):
            try:
                # This gets all mesh geometry with transforms applied
                meshes = scene.dump()
            except Exception as e:
                print(f"Warning: scene.dump() failed: {e}")
        
        # Combine all meshes into a single mesh
        if meshes:
            combined_mesh = trimesh.util.concatenate(meshes)
            processed_mesh = rescale_and_center_mesh(combined_mesh)
            processed_mesh.export(output_path, file_type='obj')
            return output_path
        else:
            raise ValueError("No valid meshes found in the scene")
    
    else:
        raise ValueError(f"Unsupported scene type: {type(scene)}")

def capture_isometric_views(obj_path):
    """
    Capture 8 isometric views of a mesh.
    
    Args:
        obj_path: Path to the processed OBJ file
        output_dir: Directory to save the images
        
    Returns:
        List of paths to captured images
    """
    init_pyvista()
    
    # Convert paths
    obj_path = pathlib.Path(obj_path)
    
    # Set up output directory
    output_dir = obj_path.parent / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define camera positions for 8 isometric views
    isometric_positions = [
        (1, 1, 1),    # Front-Right-Top
        (1, 1, -1),   # Front-Right-Bottom
        (1, -1, 1),   # Front-Left-Top
        (1, -1, -1),  # Front-Left-Bottom
        (-1, 1, 1),   # Back-Right-Top
        (-1, 1, -1),  # Back-Right-Bottom
        (-1, -1, 1),  # Back-Left-Top
        (-1, -1, -1)  # Back-Left-Bottom
    ]
    
    up_vector = (0, 0, 1)  # Consistent up vector

    # Load the mesh
    mesh = pyvista.read(str(obj_path))
    
    # Check for texture file in the same directory
    mesh_dir = os.path.dirname(obj_path)
    texture_path = None
    
    # Common texture file names to check
    texture_filenames = [
        "texture.png", "texture.jpg", "material_0.png", "material_0.jpg", "texture_0.png", "texture_0.jpg",
        os.path.basename(obj_path).replace(".obj", ".png"),
        os.path.basename(obj_path).replace(".obj", ".jpg"),
        os.path.basename(obj_path).replace(".obj", "_texture.png"),
        os.path.basename(obj_path).replace(".obj", "_texture.jpg"),
        os.path.basename(obj_path).replace(".obj", "_texture_0.png"),
        os.path.basename(obj_path).replace(".obj", "_texture_0.jpg"),
        os.path.basename(obj_path).replace(".obj", "_texture0.png"),
        os.path.basename(obj_path).replace(".obj", "_texture0.jpg")
    ]
    
    # Directories to check for textures
    texture_dirs = [
        mesh_dir,
        os.path.join(mesh_dir, "textures"),
        os.path.join(mesh_dir, "texture"),
        os.path.join(mesh_dir, "materials"),
        os.path.join(mesh_dir, ".." , "textures"),
        os.path.join(mesh_dir, ".." , "texture"),
        os.path.join(mesh_dir, ".." , "materials")
    ]
    # Search for texture files in all potential directories
    for texture_dir in texture_dirs:
        if not os.path.exists(texture_dir):
            continue
            
        for fname in texture_filenames:
            potential_texture = os.path.join(texture_dir, fname)
            if os.path.exists(potential_texture):
                texture_path = potential_texture
                print(f"Found texture: {texture_path}")
                break
                
        if texture_path:
            break
    
    # Load texture if found
    texture = None
    if texture_path:
        try:
            texture = pyvista.read_texture(str(texture_path))
        except Exception as e:
            print(f"Warning: Failed to load texture {texture_path}: {e}")
    # Load the mesh
    mesh = pyvista.read(str(obj_path))
    
    # Calculate mesh center and size for camera positioning
    center = np.array(mesh.center)
    bounds = np.array(mesh.bounds)
    sizes = bounds.reshape(3, 2)[:, 1] - bounds.reshape(3, 2)[:, 0]
    max_size = float(np.max(sizes))
    camera_distance = max_size * 2.5  # Adjust as needed
    
    image_paths = []
    
    # Capture each view
    for i, position in enumerate(isometric_positions):
        # Set up plotter
        plotter = pyvista.Plotter(off_screen=True, window_size=(1024, 1024))
        plotter.background_color = 'white'
        # Add mesh with texture if available
        try:
            if texture:
                plotter.add_mesh(mesh, texture=texture, show_edges=False)
            else:
                plotter.add_mesh(mesh, show_edges=False)
        except Exception as e:
            print(f"Warning: Failed to add mesh with texture: {e}")
            # Fallback to adding mesh without texture
            try:
                plotter.add_mesh(mesh, show_edges=False)
            except Exception as e2:
                print(f"Critical error: Failed to add mesh without texture: {e2}")
                continue
            
        # Normalize position vector and set camera distance
        position = np.array(position, dtype=float)
        position = position / np.linalg.norm(position) * camera_distance
        
        # Set camera position
        camera_position = center + position
        plotter.camera.position = tuple(camera_position)
        plotter.camera.focal_point = tuple(center)
        plotter.camera.up = up_vector
        
        # Ensure the entire mesh is visible
        plotter.camera.zoom(0.8)
        
        # Save the image
        output_path = output_dir / f"isometric_view_{i}.png"
        plotter.show(auto_close=False)
        plotter.screenshot(str(output_path))
        plotter.close()
        
        image_paths.append(str(output_path))
    
    return image_paths

  
    
def check_mesh_category(image_paths, text_prompts, threshold):
    """
    Check if any of the mesh images match the specified text prompts.
    
    Args:
        image_paths: List of paths to the isometric view images
        text_prompts: List of text prompts to match against
        
    Returns:
        tuple: (is_matched, matched_category)
    """
    # Initialize the feature matcher with default settings
    matcher = FeatureMatcher(
        encoder="DinoV2Encoder",
        gsam_box_threshold=0.6,
        gsam_text_threshold=0.6,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=False
    )
    img_path = pathlib.Path(image_paths[0])
    # Set up output directory
    output_dir = img_path.parent / "match"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_match = None
    best_score = 0
    
    # Track best matches for each image
    image_best_matches = {}
    
    # Check each image against each prompt
    for image_path in image_paths:
        img_best_prompt = None
        img_best_score = 0
        img_best_data = None
        
        # Each prompt is tried on the current image
        for prompt in text_prompts:
            try:
                # Load the image
                image_source, image = load_image(image_path)
                
                # Predict the bounding boxes
                boxes, logits, phrases = matcher.gsam.predict_boxes(img=image, caption=f"{prompt}.")
                # Calculate score based on boxes and logits
                if boxes is not None and len(boxes) > 0:
                    # Use the highest logit as the confidence score
                    current_score = max(logits)
                    
                    # Update image's best match if current score is higher
                    if current_score > img_best_score:
                        img_best_score = current_score
                        img_best_prompt = prompt
                        img_best_data = (image_source, boxes, logits, phrases)
                    
                    # Update global best match if current score is higher
                    if current_score > best_score:
                        best_score = current_score
                        best_match = prompt
                
            except Exception as e:
                print(f"Error processing {prompt} on {image_path}: {e}")
                continue
        
        # Save the best match for this image
        if img_best_prompt and img_best_data:
            image_best_matches[image_path] = (img_best_prompt, img_best_score)
            
            # Extract the saved data
            image_source, boxes, logits, phrases = img_best_data
            
            # Create annotated image
            annotated_frame = annotate(
                image_source=image_source, 
                boxes=boxes, 
                logits=logits, 
                phrases=phrases
            )
            
            # Save annotated image with informative filename
            base_name = os.path.basename(image_path).split('.')[0]
            
            save_path = os.path.join(output_dir, f"{base_name}_match_{img_best_prompt}_{img_best_score:.2f}.png")
            cv2.imwrite(save_path, annotated_frame)
    
    
    if best_score > threshold and best_match is not None:
        return True, best_match
    else:
        return False, None
    
def check_mesh_category_light(image_paths, text_prompts, threshold, obj_path):
    """
    Check if any of the mesh images match the specified text prompts.
    
    Args:
        image_paths: List of paths to the isometric view images
        text_prompts: List of text prompts to match against
        
    Returns:
        tuple: (is_matched, matched_category)
    """  
    # Initialize the feature matcher with default settings
    matcher = FeatureMatcher(
        encoder="DinoV2Encoder",
        gsam_box_threshold=threshold,
        gsam_text_threshold=threshold,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=False
    )
    
    # Check each image against each prompt
    for image_path in image_paths:
        
        # Each prompt is tried on the current image
        for prompt in text_prompts:
            # Load the image
            _, image = load_image(image_path)
            
            # Predict the bounding boxes
            boxes, logits, phrases = matcher.gsam.predict_boxes(img=image, caption=f"{prompt}.")
            # Calculate score based on boxes and logits
            if boxes is not None and len(boxes) > 0:
                # Use the highest logit as the confidence score
                current_score = max(logits)
                if current_score >= threshold:
                    # # Removes images and .obj file
                    # img_path = pathlib.Path(image_paths[0])
                    # grandparent_dir = img_path.parent.parent  # Parent directory to keep
                    
                    # # Delete all files in the grandparent directory
                    # for item in grandparent_dir.glob('*'):
                    #     if item.is_file():
                    #         item.unlink()  # Delete files
                    #     elif item.is_dir():
                    #         shutil.rmtree(item)  # Delete subdirectories
                    
                    return True, prompt
                
    # # Delete all files in the grandparent directory
    # img_path = pathlib.Path(image_paths[0])
    # grandparent_dir = img_path.parent.parent  # Parent directory to keep
    
    # # Delete all files in the grandparent directory
    # for item in grandparent_dir.glob('*'):
    #     if item.is_file():
    #         item.unlink()  # Delete files
    #     elif item.is_dir():
    #         shutil.rmtree(item)  # Delete subdirectories
            
    return False, None             
    
@click.command()
@click.option('--asset-path', required=True, help='Path to the 3D model file (.obj, .glb, .gltf, .dae)')
@click.option('--text-prompts', default="closet,cabinet,drawer,refrigerator,table", 
              help='Comma-separated list of prompts to match against')
@click.option('--light-mode', is_flag = True)

def filter_mesh(asset_path, text_prompts, light_mode):
    """
    Process a 3D model file and determine if it matches any of the specified furniture categories.
    """
    try:
        # # Convert text_prompts string to list
        prompt_list = [p.strip() for p in text_prompts.split(',')]
        
        # Process the mesh
        processed_mesh_path = process_mesh(asset_path)
        
        # Capture isometric views
        image_paths = capture_isometric_views(processed_mesh_path)
        
        if light_mode:
            is_match, matched_category = check_mesh_category_light(image_paths, prompt_list, threshold = 0.85, obj_path = processed_mesh_path)
            
        else:
            # Check if the mesh matches any of the categories
            is_match, matched_category = check_mesh_category(image_paths, prompt_list, threshold = 0.85)
            
        # # Print result
        result = {
            "is_match": is_match,
            "matched_category": matched_category
        }
        print(result) 
    
        return result
    
    except Exception as e:
        print(f"Error: {e}")
        return {"is_match": False, "matched_category": None}

if __name__ == '__main__':
    filter_mesh()