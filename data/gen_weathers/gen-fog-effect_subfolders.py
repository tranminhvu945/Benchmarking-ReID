#!/usr/bin/env python
import os
import subprocess
from pathlib import Path
import argparse

def get_image_files(directory):
    """Get all image files in directory"""
    image_extensions = {'.jpg', '.png'}
    return [f for f in Path(directory).iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions]

def get_unprocessed_images(image_files, depth_dir, fog_dir):
    """Get list of images that haven't been fully processed yet"""
    unprocessed = []
    skipped = 0
    for img_path in image_files:
        depth_path = Path(depth_dir) / f"{img_path.stem}.npy"
        fog_path = Path(fog_dir) / f"{img_path.stem}.jpg"
        
        # Check if both depth and fog files exist
        if not (depth_path.exists() and fog_path.exists()):
            unprocessed.append(img_path)
        else:
            skipped += 1
            
    return unprocessed, skipped

def run_midas_and_fog(image_dir, depth_dir, fog_dir, gpu_ids):
    """Run MiDaS and Fog Effect Generator on a directory"""
    Path(depth_dir).mkdir(parents=True, exist_ok=True)
    Path(fog_dir).mkdir(parents=True, exist_ok=True)
    
    # Check for images
    image_files = get_image_files(image_dir)
    if not image_files:
        print(f"No image files found in {image_dir}")
        return False
    
    # Get unprocessed images
    unprocessed_files, skipped_count = get_unprocessed_images(image_files, depth_dir, fog_dir)
    
    if skipped_count > 0:
        print(f"\nSkipping {skipped_count} already processed images")
    
    if not unprocessed_files:
        print(f"All images in {image_dir} have been processed")
        return True
    
    print(f"\nFound {len(unprocessed_files)} unprocessed images in {image_dir}")
    
    # Create temporary directory for unprocessed images
    temp_dir = Path(image_dir) / ".temp_processing"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Create symbolic links for unprocessed images
        for img_path in unprocessed_files:
            (temp_dir / img_path.name).symlink_to(img_path)
        
        # Run MiDaS on unprocessed images
        print(f"Running MiDaS on: {image_dir}")
        midas_command = [
            "python", "/home/uit2023/LuuTru/Vutm/dataset/gen_rain/MiDaS_Depth_Estimation_goc.py",
            "--img_path", str(temp_dir),
            "--save_folder", str(depth_dir),
            "--midas_model", "DPT_Large",
            "--use_cuda",
            "--gpu_ids", gpu_ids
        ]
        
        result = subprocess.run(midas_command)
        if result.returncode != 0:
            print(f"MiDaS failed for directory: {image_dir}")
            return False
        
        # Verify depth files were generated
        depth_files = [Path(depth_dir) / f"{img.stem}.npy" for img in unprocessed_files]
        missing_depth = [f for f in depth_files if not f.exists()]
        if missing_depth:
            print(f"Missing depth files: {missing_depth}")
            return False
        
        print(f"Generated {len(depth_files)} depth files")
        
        # Run Fog Effect Generator
        print(f"Running Fog Effect Generator on: {image_dir}")
        fog_command = [
            "python", "/home/uit2023/LuuTru/Vutm/dataset/gen_rain/Fog_Effect_Generator.py",
            "--clear_path", str(temp_dir),
            "--depth_path", str(depth_dir),
            "--save_folder", str(fog_dir),
            "--gpu_ids", gpu_ids
        ]
        
        result = subprocess.run(fog_command)
        if result.returncode != 0:
            print(f"Fog Effect Generator failed for directory: {image_dir}")
            return False
        
        # Verify fog effect files were generated
        fog_files = [Path(fog_dir) / f"{img.stem}.jpg" for img in unprocessed_files]
        missing_fog = [f for f in fog_files if not f.exists()]
        if missing_fog:
            print(f"Missing fog effect files: {missing_fog}")
            return False
        
        print(f"Generated {len(fog_files)} fog effect files")
        return True
        
    finally:
        # Cleanup: remove temporary directory
        if temp_dir.exists():
            for link in temp_dir.iterdir():
                link.unlink()
            temp_dir.rmdir()

def process_directory_structure(base_image_dir, base_depth_dir, base_fog_dir, gpu_ids):
    """Process entire directory structure"""
    success_count = 0
    fail_count = 0
    
    # Process base directory first if it contains images
    image_files = get_image_files(base_image_dir)
    if image_files:
        print(f"\nProcessing base directory: {base_image_dir}")
        if run_midas_and_fog(base_image_dir, base_depth_dir, base_fog_dir, gpu_ids):
            success_count += 1
        else:
            fail_count += 1
    
    # Process subdirectories
    for image_dir in Path(base_image_dir).rglob("*/"):
        if not image_dir.is_dir() or image_dir.name.startswith('.'):
            continue
        
        # Get relative path and create corresponding output directories
        relative_path = image_dir.relative_to(base_image_dir)
        depth_dir = Path(base_depth_dir) / relative_path
        fog_dir = Path(base_fog_dir) / relative_path
        
        # Check for images in directory
        if not get_image_files(image_dir):
            print(f"\nNo images in {image_dir}, skipping...")
            continue
        
        print(f"\nProcessing directory: {image_dir}")
        print(f"Depth directory: {depth_dir}")
        print(f"Fog effect directory: {fog_dir}")
        
        if run_midas_and_fog(image_dir, depth_dir, fog_dir, gpu_ids):
            success_count += 1
        else:
            fail_count += 1
    
    return success_count, fail_count

def main():
    parser = argparse.ArgumentParser(description='Run MiDaS and Fog Effect Generator recursively')
    parser.add_argument('--gpu_ids', type=str, required=True, help='GPU IDs to use (e.g., "0" or "0,1")')
    parser.add_argument('--base_image_path', type=str, 
                       default="/storageStudents/ncsmmlab/tungufm/UIT-Script/gen-rain/images",
                       help='Base path for input images')
    parser.add_argument('--depth_save_folder', type=str,
                       default="/storageStudents/ncsmmlab/tungufm/UIT-Script/datasets/UIT-ADrone/depth-files-rain",
                       help='Base path for saving depth files')
    parser.add_argument('--fog_save_folder', type=str,
                       default="/storageStudents/ncsmmlab/tungufm/UIT-Script/gen-rain/rain-effect-images",
                       help='Base path for saving rain effect images')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    base_image_dir = Path(args.base_image_path).resolve()
    base_depth_dir = Path(args.depth_save_folder).resolve()
    base_fog_dir = Path(args.fog_save_folder).resolve()
    
    if not base_image_dir.exists():
        raise ValueError(f"Image directory does not exist: {base_image_dir}")
        
    if not base_image_dir.is_dir():
        raise ValueError(f"Image path must be a directory: {base_image_dir}")
    
    print("\nStarting processing with configuration:")
    print(f"Base image directory: {base_image_dir}")
    print(f"Base depth directory: {base_depth_dir}")
    print(f"Base fog effect directory: {base_fog_dir}")
    print(f"GPU IDs: {args.gpu_ids}")
    
    success_count, fail_count = process_directory_structure(
        base_image_dir,
        base_depth_dir,
        base_fog_dir,
        args.gpu_ids
    )
    
    print("\nProcessing completed!")
    print(f"Successfully processed directories: {success_count}")
    print(f"Failed directories: {fail_count}")
    print(f"Total directories attempted: {success_count + fail_count}")

if __name__ == '__main__':
    main()