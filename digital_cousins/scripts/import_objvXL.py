import os
import objaverse
import objaverse.xl as oxl
import pandas as pd
import shutil
import random
import string
import click
import time
import logging
from typing import Dict, Hashable, Any, Set, Optional
from datetime import datetime
from pathlib import Path
from collections import deque
import subprocess
import ast
from omnigibson.examples.objects.import_custom_object import import_custom_object
from omnigibson.macros import gm

BASE_DIR = os.path.join(os.path.dirname(__file__),"..", "..","deps", "Objaverse")
# SOURCE = "thingiverse"  # ["github", "thingiverse", "sketchfab", "smithsonian"]
GITHUB_SAVED_FORMAT = "files"  # ["zip", "tar", "tar.gz", "files"]
DOWNLOAD_BATCH_SIZE = 1  # Number of files to download in each batch
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
DOWNLOAD_TIMEOUT = 300  # 5 minutes timeout for downloads
SEMANTIC_DICT = "closet,cabinet,drawer,refrigerator,oven,microwave,table"

# # Disable swap temporarily
# sudo swapoff -a
# # Re-enable later if needed
# sudo swapon -a

class DownloadTracker:

    def __init__(self,max_batch_size=1000):
        self.current_working_dir = os.getcwd()
        self.logs_dir = os.path.join(BASE_DIR, "download_logs")
        self.used_model_names: Set[str] = set()
        self.format_to_log = {}
        self.initialize_logs()
        self.pending_logs = deque(maxlen=max_batch_size)
        self.max_batch_size = max_batch_size
    
    def get_log_path(self, file_format: str) -> str:
        """Get the path for a specific format's log file"""

        return os.path.join(self.logs_dir, f"download_log_{file_format.lstrip('.')}.csv")
    
    def initialize_logs(self):
        """Initialize log directory and load existing model names"""

        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Load all existing model names from all log files
        for file in os.listdir(self.logs_dir):
            if file.startswith("download_log_") and file.endswith(".csv"):
                log_path = os.path.join(self.logs_dir, file)
                try:
                    df = pd.read_csv(log_path)
                    if 'model_name' in df.columns:
                        self.used_model_names.update(
                            name for name in df['model_name'].dropna().unique()
                        )
                except Exception:
                    continue

    def generate_unique_model_name(self) -> str:
        """Generate a unique 6-letter model name"""

        while True:
            model_name = ''.join(random.choices(string.ascii_uppercase, k=6))
            if model_name not in self.used_model_names:
                self.used_model_names.add(model_name)
                return model_name
    
    def get_format_log(self, file_format: str, force_refresh: bool = True) -> pd.DataFrame:
        """Get or create log DataFrame for specific format"""

        if force_refresh or file_format not in self.format_to_log:
            log_path = self.get_log_path(file_format)
            if os.path.exists(log_path):
                self.format_to_log[file_format] = pd.read_csv(log_path)
            else:
                self.format_to_log[file_format] = pd.DataFrame(columns=[
                    'timestamp', 'file_identifier', 'format', 'download_status',
                    'model_name', 'category', 'source', 'local_path', 'sha256'
                ])
        return self.format_to_log[file_format]
    
    def is_file_processed(self, sha256: str, file_format: str) -> bool:
        """Check if file was previously processed"""

        log_df = self.get_format_log(file_format)
        return sha256 in log_df['sha256'].values
    
    def log_download(self, 
                     file_identifier: str, 
                     file_format: str, 
                     status: str,
                     model_name: str = None,
                     category: str = None,
                     source: str = None,
                     local_path: str = None,
                     sha256: str = None):
        """Log download attempt to format-specific CSV"""

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'file_identifier': file_identifier,
            'format': file_format,
            'download_status': status,
            'model_name': model_name,
            'category': category,
            'source': source,
            'local_path': local_path,
            'sha256': sha256
        }
        
        log_df = self.get_format_log(file_format)
        log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
        self.format_to_log[file_format] = log_df
        log_path = self.get_log_path(file_format)
        log_df.to_csv(log_path, index=False)
        
    def update_status(self, file_identifier: str, file_format: str, new_status: str, category:str, local_path: str):
        """
        Update the status of a specific file in the log.
        
        Args:
            file_identifier: The identifier of the file
            file_format: The format of the file
            new_status: The new status to set
        """

        log_df = self.get_format_log(file_format)
        mask = log_df['file_identifier'] == file_identifier
        if any(mask):
            log_df.loc[mask, 'download_status'] = new_status
            log_df.loc[mask, 'category'] = category
            log_df.loc[mask, 'local_path'] = local_path
            self.format_to_log[file_format] = log_df
            # Save immediately to prevent loss of status on crashes
            log_path = self.get_log_path(file_format)
            log_df.to_csv(log_path, index=False)

    def save_all_logs(self):
        """Save all format logs to their respective files"""

        for file_format, log_df in self.format_to_log.items():
            log_path = self.get_log_path(file_format)
            log_df.to_csv(log_path, index=False)

download_tracker = DownloadTracker()

def find_file_in_source_dir(filename: str) -> Optional[str]:
    """
    Find a file within the specific source directory.
    
    Args:
        filename: Name of the file to find
    Returns:
        Full path to the file if found, None otherwise
    """
    if SOURCE == "github":
        source_dir = os.path.join(BASE_DIR, SOURCE.lower())
    elif SOURCE == "sketchfab":
        source_dir = BASE_DIR
        
    if not os.path.exists(source_dir):
        return None
        
    for root, _, files in os.walk(source_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

def handle_missing_object(
    file_identifier: str,
    sha256: str,
    metadata: Dict[Hashable, Any].clear
    ) -> None:
        print("\n\n\n---HANDLE_MISSING_OBJECT CALLED---\n",
            f"  {file_identifier=}\n {metadata=}\n\n\n")
        
        # file_format = os.path.splitext(local_path)[1]
        
        # download_tracker.log_download(
        #     file_identifier=file_identifier,
        #     file_format=file_format,
        #     status='missing',
        # )

def handle_found_object(
    local_path: str,
    file_identifier: str,
    sha256: str,
    metadata: Dict[Hashable, Any]
    ) -> None:
        print("\n\n\n---HANDLE_FOUND_OBJECT CALLED---\n",
            f"  {local_path=}\n  {file_identifier=}\n {metadata=}\n\n\n")
        file_format = os.path.splitext(local_path)[1]
        if not download_tracker.is_file_processed(sha256, file_format):
            model_name = download_tracker.generate_unique_model_name()

            download_tracker.log_download(
                file_identifier=file_identifier,
                file_format=file_format,
                status='found',
                model_name=model_name,
                local_path=None,
                source=SOURCE,
                sha256=sha256,
            )

# def handle_modified_object(
#     local_path: str,
#     file_identifier: str,
#     new_sha256: str,
#     old_sha256: str,
#     metadata: Dict[Hashable, Any],
#     ) -> None:
#         print("\n\n\n---HANDLE_MODIFIED_OBJECT CALLED---\n",
#               f"  {local_path=}\n  {file_identifier=}\n  {old_sha256=}\n  {new_sha256}\n  {metadata=}\n\n\n")

def handle_new_object(
    local_path: str,
    file_identifier: str,
    sha256: str,
    metadata: Dict[Hashable, Any]
    ) -> None:
        print("\n\n\n---HANDLE_NEW_OBJECT CALLED---\n")
            #   f"  {local_path=}\n  {file_identifier=}\n {metadata=}\n\n\n")
        file_format = os.path.splitext(local_path)[1]
        
        if not download_tracker.is_file_processed(sha256, file_format):
            model_name = download_tracker.generate_unique_model_name()
            
            download_tracker.log_download(
                file_identifier=file_identifier,
                file_format=file_format,
                status='found',
                model_name=model_name,
                local_path=None,
                source=SOURCE,
                sha256=sha256,
            )
            
def filter_annotations(annotations: pd.DataFrame, target_format: str) -> pd.DataFrame:
    """
    Filter annotations based on file format and source.
    
    Args:
        annotations: Original annotations DataFrame
        target_format: Desired file format (e.g., '.obj')
    
    Returns:
        Filtered DataFrame containing only entries matching the criteria
    """
    # Convert target format to lowercase for case-insensitive comparison
    target_format = target_format.lower()[1:]
    # Create mask for unprocessed files
    log_df = download_tracker.get_format_log(target_format)
    processed_files = set(log_df['sha256'].values)
    # Filter by source and file format
    mask = (
        (annotations['source'] == SOURCE) &
        (annotations['fileType'] == target_format)&
        (~annotations['sha256'].isin(processed_files))
    )
    
    filtered = annotations[mask]
    return filtered

def get_file_name(fileIdentifier: str):
    if SOURCE == "github":
        filename = os.path.basename(fileIdentifier)
    elif SOURCE == "sketchfab":
        filename = os.path.basename(fileIdentifier)+ ".glb"
    return filename
    
def download_3d_objects(annotations, download_dir, target_format: str):
    """Download 3D objects in batches with error handling."""
    os.makedirs(download_dir, exist_ok=True)
    total_files = len(annotations)
    if total_files == 0:
        print(f"No files found from {SOURCE} with format {target_format}")
        return
    retry_count = 0
    
    # Process in batches
    for i in range(0, total_files, DOWNLOAD_BATCH_SIZE):
        while retry_count < MAX_RETRIES:
            batch = annotations.iloc[i:i+DOWNLOAD_BATCH_SIZE]
            print(f"Processing batch {i//DOWNLOAD_BATCH_SIZE + 1}/{(total_files+DOWNLOAD_BATCH_SIZE-1)//DOWNLOAD_BATCH_SIZE} in download_3d_objects in attempt {retry_count}")
             
            try:
                oxl.download_objects(
                    objects=batch,
                    download_dir=download_dir,
                    handle_missing_object=handle_missing_object,
                    handle_found_object=handle_found_object,
                    # handle_modified_object=handle_modified_object,
                    handle_new_object = handle_new_object,
                    save_repo_format=GITHUB_SAVED_FORMAT
                )
                return
             
            except Exception as e:
                retry_count += 1
                if retry_count < MAX_RETRIES:
                    logging.warning(f"Download attempt {retry_count} failed: {str(e)}")
                    time.sleep(RETRY_DELAY * retry_count)  # Exponential backoff
                else:
                    logging.error(f"All download attempts failed")
                    # Log failed batch
                    for _, row in batch.iterrows():
                        file_format = os.path.splitext(row['fileIdentifier'])[1]
                        download_tracker.log_download(
                            file_identifier=row['fileIdentifier'],
                            file_format=file_format,
                            status='failed',
                        )
                
                # print(f"Error in batch {i//DOWNLOAD_BATCH_SIZE + 1}: {e}")
                # # Log failed batch
                # for _, row in batch.iterrows():
                #     file_format = os.path.splitext(row['fileIdentifier'])[1]
                #     download_tracker.log_download(
                #         file_identifier=row['fileIdentifier'],
                #         file_format=file_format,
                #         status='failed',
                #     )
                # continue
            
def delete_repo(path: str):
    if SOURCE == "github":
        """Delete the whole github repo"""
        try:
            path_parts = path.split(os.sep)
            try:
                repos_index = path_parts.index('repos')
            except ValueError:
                print("Error: Path does not contain 'repos' directory")
                return
            except IndexError:
                print("Error: No repository name found after 'repos' directory")
                return
                
            # Construct the path to the repository directory
            repo_path = os.sep.join(path_parts[:repos_index + 2])  # Include up to repo name
            
            # Check if directory exists
            if os.path.exists(repo_path):
                # Remove the directory and all its contents
                shutil.rmtree(repo_path)
                print(f"Successfully deleted repository directory: {repo_path}")
            # else:
            #     print(f"Directory does not exist: {repo_path}")
                
        except Exception as e:
            print(f"Error deleting repository: {str(e)}")
            
    elif SOURCE == "sketchfab":
        try:
            os.remove(path)
            print(f"Successfully removed {path}")
        except FileNotFoundError:
            print(f"File not found: {path}")
        

@click.command()
@click.option('--target-format', '-f', required=True, type = click.STRING, help='File format to process (e.g., .obj)')
@click.option('--sample-size', '-s', default=10, type = click.INT, help='Total number of objects to process')
@click.option('--batch-size', '-b', default=1,  type = click.INT, help='Number of objects to process in each batch')
@click.option('--get-all', '-ga', is_flag=True, help='Whether to download all files in target-format')
@click.option('--headless', '-hl', is_flag = True, help='Whether to download in headless mode')

def main(target_format: str, sample_size: int, batch_size: int, get_all:bool, headless:bool):
    """
    Process Objaverse objects in batches with automatic cleanup.
    
    This function handles the main workflow:
    1. Fetches annotations from Objaverse
    2. Filters objects based on format
    3. Downloads objects in batches
    4. Processes downloaded files
    5. Cleans up after processing
    
    Args:
        target_format (str): File format to process (e.g., '.obj')
        sample_size (int): Total number of objects to process
        batch_size (int): Number of objects to process in each batch
        get_all (bool): Whether to process all available files
    """
    try:
        global SOURCE  
        # Set SOURCE based on target_format
        if target_format.lower() in ['.glb']:
            SOURCE = "sketchfab" # or "smithsonian"
        elif target_format.lower() in ['.stl']:
            SOURCE = "thingiverse"
        else:
            SOURCE = "github"  # default value
            
        print(f"Selected source: {SOURCE} for format: {target_format}")
        print("Fetching annotations...")
        annotations = oxl.get_annotations(download_dir=BASE_DIR)
        
        # Filter and sample annotations
        # filtered_annotations = annotations[annotations['fileType'] == target_format.lower()]
        filtered_annotations = filter_annotations(annotations, target_format)
        if not get_all:
            filtered_annotations = filtered_annotations.sample(min(sample_size, len(filtered_annotations)))
        print(f"Selected {len(filtered_annotations)} objects for processing")
        print(f"Source distribution:\n{filtered_annotations['source'].value_counts()}")
        
        # Process in batches
        total_batches = (len(filtered_annotations) + batch_size - 1) // batch_size
        
        if headless:
            gm.HEADLESS = True

        for batch_num in range(total_batches):
            print(f"\nProcessing batch {batch_num + 1}/{total_batches}")
            
            # Get current batch of annotations
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(filtered_annotations))
            batch_annotations = filtered_annotations.iloc[start_idx:end_idx]
            # Download current batch
            download_3d_objects(batch_annotations, BASE_DIR, target_format)
            
            # Process downloaded files in current batch
            log_df = download_tracker.get_format_log(target_format, force_refresh=True)
            
            # Filter for successful downloads in current batch
            successful_downloads = log_df[(log_df['download_status'] == 'found')]
            
            print(f"Processing {len(successful_downloads)} successfully downloaded files in batch {batch_num + 1}")
            current_repo_path = []

            # Process each file in the batch
            for _, row in successful_downloads.iterrows():
                if pd.isna(row['model_name']):
                    continue
                    
                # Find actual file path
                filename = get_file_name(row['file_identifier'])
                file_path = find_file_in_source_dir(filename)
                current_repo_path.append(file_path)
                if file_path and os.path.exists(file_path):
                    
                    try:
                        print("Filtering object based on semantic...")

                        # Filter out the mesh that are not in the list of target
                        cmd = ["python", "filter_objv.py", "--asset-path", file_path, "--text-prompts", SEMANTIC_DICT, "--light-mode"]
                        result = subprocess.run(cmd, capture_output=True, text=True)

                        # Extract the result dictionary from the output
                        # Get the last non-empty line
                        last_line = [line.strip() for line in result.stdout.split('\n') if line.strip()][-1]
                        
                        # Parse the Python dictionary
                        result_dict = ast.literal_eval(last_line)
                        
                        # Extract values
                        is_match = result_dict["is_match"]
                        matched_category = result_dict["matched_category"]

                        # Now use the values in your code
                        if is_match:
                            print(f"✅ Object matches category: {matched_category}")
                            # Your code for matching objects...
                            
                        else:
                            print("❌ Object does not match any semantic categories")
                            download_tracker.update_status(
                                file_identifier=row['file_identifier'],
                                file_format=target_format,
                                local_path=file_path,
                                new_status='filtered',
                                category = None,
                            )
                            continue  # Skip to next object
                        
                        # Import the object
                        import_custom_object.callback(
                            asset_path=file_path,
                            category="objaverse",
                            model=row['model_name'],
                            collision_method="coacd",
                            hull_count=32,
                            up_axis="z",
                            headless=headless,
                            scale= 1,
                            check_scale = True,
                            rescale = True,
                            overwrite=True,
                            n_submesh = 20,
                        )
                        
                        # Update status and cleanup
                        download_tracker.update_status(
                            file_identifier=row['file_identifier'],
                            file_format=target_format,
                            local_path=file_path,
                            new_status='converted',
                            category = matched_category,
                        )
                        print(f" ************** Successfully imported {row['file_identifier']} as {row['model_name']} ****************")
                        
                    except Exception as e:
                        print(f"Error processing {row['file_identifier']}: {e}")
                        download_tracker.update_status(
                            file_identifier=row['file_identifier'],
                            file_format=target_format,
                            local_path=file_path,
                            new_status='failed',
                            category=None,
                        )
                else:
                    print(f"Could not find file path for {row['file_identifier']}")
            
            print(f"Completed batch {batch_num + 1}")
            # Delete the repository after successful processing
            for repo_path in current_repo_path:
                delete_repo(repo_path)  
            
    except Exception as e:
        print(f"Error in main process: {e}")
    finally:
        download_tracker.save_all_logs()

if __name__ == "__main__":
    main()
