#!/usr/bin/env python3
"""
Single Object Tracking CLI
Track a specific object selected by bounding box or point click
Output: White mask video showing only the selected object
"""

import os
import sys
import cv2
import torch
import numpy as np
import argparse
import gc
from PIL import Image
import imageio

# Add project paths
sys.path.append(".")
sys.path.append("./sam")

# Import core modules
from SimpleSegTracker import SimpleSegTracker as SegTracker
from model_args import aot_args, sam_args, segtracker_args

def get_first_frame_selection(video_path, selection_type, selection_params):
    """
    Get the first frame and perform object selection
    
    Args:
        video_path: Path to input video
        selection_type: 'bbox' or 'point'
        selection_params: bbox=[x1,y1,x2,y2] or point=[x,y]
    
    Returns:
        first_frame: numpy array (h,w,3)
        mask: numpy array (h,w) with selected object as 1
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError("Could not read first frame from video")
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Initialize SAM for interactive segmentation
    sam_args_local = sam_args.copy()
    sam_args_local['generator_args'] = {
        'points_per_side': 32,
        'pred_iou_thresh': 0.88,
        'stability_score_thresh': 0.95,
        'crop_n_layers': 1,
        'crop_n_points_downscale_factor': 2,
        'min_mask_region_area': 100,
    }
    
    # Create a minimal segtracker just for first frame selection
    segtracker_args_temp = {'sam_gap': 1, 'min_area': 100, 'max_obj_num': 1, 'min_new_obj_iou': 0.8}
    aot_args_temp = aot_args.copy()
    
    segtracker = SegTracker(segtracker_args_temp, sam_args_local, aot_args_temp)
    
    if selection_type == 'bbox':
        # selection_params = [x1, y1, x2, y2]
        bbox = np.array([[selection_params[0], selection_params[1]], 
                        [selection_params[2], selection_params[3]]])
        mask, _ = segtracker.seg_acc_bbox(frame_rgb, bbox)
        
    elif selection_type == 'point':
        # selection_params = [x, y]
        coords = np.array([[selection_params[0], selection_params[1]]])
        modes = np.array([1])  # positive point
        mask, _ = segtracker.seg_acc_click(frame_rgb, coords, modes, multimask=False)
    
    else:
        raise ValueError("selection_type must be 'bbox' or 'point'")
    
    # Clean up
    del segtracker
    torch.cuda.empty_cache()
    gc.collect()
    
    # Return only the selected object mask (binary)
    binary_mask = (mask > 0).astype(np.uint8)
    
    return frame_rgb, binary_mask

def track_single_object(video_path, first_frame, object_mask, output_dir):
    """
    Track the selected object through the entire video
    
    Args:
        video_path: Path to input video
        first_frame: First frame (h,w,3)
        object_mask: Binary mask of selected object (h,w)
        output_dir: Output directory
    
    Returns:
        mask_list: List of masks for each frame
        num_frames: Total number of processed frames
    """
    print("Initializing tracker for single object...")
    
    # Configure for single object tracking
    segtracker_args_local = {
        'sam_gap': 9999,  # Don't run SAM again after first frame
        'min_area': 50,
        'max_obj_num': 1,
        'min_new_obj_iou': 0.8,
    }
    
    sam_args_local = sam_args.copy()
    aot_args_local = aot_args.copy()
    
    # Initialize tracker
    segtracker = SegTracker(segtracker_args_local, sam_args_local, aot_args_local)
    segtracker.restart_tracker()
    
    # Add the selected object as reference
    segtracker.add_reference(first_frame, object_mask)
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing {total_frames} frames at {fps} FPS...")
    
    mask_list = []
    frame_idx = 0
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    mask_dir = os.path.join(output_dir, 'masks')
    os.makedirs(mask_dir, exist_ok=True)
    
    with torch.cuda.amp.autocast():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if frame_idx == 0:
                # First frame - use the provided mask
                pred_mask = object_mask
            else:
                # Track the object
                pred_mask = segtracker.track(frame_rgb, update_memory=True)
                # Convert to binary (0 or 1)
                pred_mask = (pred_mask > 0).astype(np.uint8)
            
            # Save binary mask as image
            mask_img = (pred_mask * 255).astype(np.uint8)  # Convert to 0-255 range
            mask_pil = Image.fromarray(mask_img, mode='L')  # Grayscale
            mask_pil.save(os.path.join(mask_dir, f'{frame_idx:06d}.png'))
            
            mask_list.append(pred_mask)
            
            if frame_idx % 50 == 0:
                print(f"Processed frame {frame_idx}/{total_frames}")
                
            frame_idx += 1
            
            # Memory cleanup
            if frame_idx % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    cap.release()
    
    # Cleanup
    del segtracker
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Finished processing {frame_idx} frames")
    return mask_list, frame_idx

def create_mask_video(video_path, mask_list, output_path):
    """
    Create a video showing white masks on black background
    
    Args:
        video_path: Original video path (for fps reference)
        mask_list: List of binary masks
        output_path: Output video path
    """
    print("Creating white mask video...")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    for i, mask in enumerate(mask_list):
        # Convert binary mask to white (255) on black (0) background
        mask_frame = (mask * 255).astype(np.uint8)
        
        # Write frame
        out.write(mask_frame)
        
        if i % 50 == 0:
            print(f"Writing mask frame {i}/{len(mask_list)}")
    
    out.release()
    print(f"Saved mask video: {output_path}")

def create_overlay_video(video_path, mask_list, output_path, mask_color=(0, 255, 0), alpha=0.6):
    """
    Create a video with colored mask overlay on original video
    
    Args:
        video_path: Original video path
        mask_list: List of binary masks
        output_path: Output video path
        mask_color: RGB color for mask overlay
        alpha: Transparency of overlay
    """
    print("Creating overlay video...")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while cap.isOpened() and frame_idx < len(mask_list):
        ret, frame = cap.read()
        if not ret:
            break
        
        mask = mask_list[frame_idx]
        
        # Create colored overlay
        overlay = frame.copy()
        overlay[mask > 0] = mask_color
        
        # Blend with original frame
        result = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
        
        out.write(result)
        
        if frame_idx % 50 == 0:
            print(f"Writing overlay frame {frame_idx}/{len(mask_list)}")
            
        frame_idx += 1
    
    cap.release()
    out.release()
    print(f"Saved overlay video: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Single Object Tracking with Interactive Selection")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("-o", "--output", default="./single_track_output", help="Output directory")
    
    # Selection method (mutually exclusive)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--bbox", nargs=4, type=int, metavar=('X1', 'Y1', 'X2', 'Y2'),
                          help="Bounding box selection: x1 y1 x2 y2")
    selection.add_argument("--point", nargs=2, type=int, metavar=('X', 'Y'),
                          help="Point selection: x y")
    
    # Output options
    parser.add_argument("--mask-only", action="store_true", 
                       help="Output only white mask video (no overlay)")
    parser.add_argument("--device", default="cuda", help="Device to use (default: cuda)")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input_video):
        print(f"Error: Input video '{args.input_video}' does not exist")
        return 1
    
    # Set device
    if args.device == "cuda":
        sam_args["gpu_id"] = 0
        aot_args["gpu_id"] = 0
    elif args.device.startswith("cuda:"):
        gpu_id = int(args.device.split(":")[1])
        sam_args["gpu_id"] = gpu_id
        aot_args["gpu_id"] = gpu_id
    else:
        sam_args["gpu_id"] = args.device
        aot_args["gpu_id"] = 0
    
    print("=" * 60)
    print("Single Object Tracking with Interactive Selection")
    print("=" * 60)
    print(f"Input video: {args.input_video}")
    print(f"Output directory: {args.output}")
    
    if args.bbox:
        print(f"Selection: Bounding box {args.bbox}")
        selection_type = "bbox"
        selection_params = args.bbox
    else:
        print(f"Selection: Point {args.point}")
        selection_type = "point"
        selection_params = args.point
        
    print(f"Device: {args.device}")
    print("=" * 60)
    
    try:
        # Step 1: Get first frame and select object
        print("Step 1: Selecting object in first frame...")
        first_frame, object_mask = get_first_frame_selection(
            args.input_video, selection_type, selection_params
        )
        
        # Check if object was found
        if np.sum(object_mask) == 0:
            print("Error: No object found with the given selection. Try different coordinates.")
            return 1
            
        print(f"✓ Object selected successfully (area: {np.sum(object_mask)} pixels)")
        
        # Step 2: Track object through video
        print("Step 2: Tracking object through video...")
        mask_list, num_frames = track_single_object(
            args.input_video, first_frame, object_mask, args.output
        )
        
        # Step 3: Create output videos
        print("Step 3: Creating output videos...")
        
        # Always create white mask video
        mask_video_path = os.path.join(args.output, "mask_video.mp4")
        create_mask_video(args.input_video, mask_list, mask_video_path)
        
        # Create overlay video unless mask-only is specified
        if not args.mask_only:
            overlay_video_path = os.path.join(args.output, "overlay_video.mp4")
            create_overlay_video(args.input_video, mask_list, overlay_video_path)
        
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"Processed {num_frames} frames")
        print(f"Object area: {np.sum(object_mask)} pixels")
        print(f"Individual masks saved in: {os.path.join(args.output, 'masks')}")
        print(f"White mask video: {mask_video_path}")
        if not args.mask_only:
            print(f"Overlay video: {overlay_video_path}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())