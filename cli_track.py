#!/usr/bin/env python3
"""
CLI Video Tracking and Segmentation Tool
Based on Segment-and-Track-Anything (SAM-Track)
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
from aot_tracker import _palette

def save_prediction(pred_mask, output_dir, file_name):
    """Save prediction mask as PNG with color palette"""
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask.save(os.path.join(output_dir, file_name))

def colorize_mask(pred_mask):
    """Convert mask to RGB for visualization"""
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)

def draw_mask(img, mask, alpha=0.5):
    """Draw mask overlay on image"""
    from scipy.ndimage import binary_dilation
    
    img_mask = img.copy()
    binary_mask = (mask != 0)
    countours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
    foreground = img * (1 - alpha) + colorize_mask(mask) * alpha
    img_mask[binary_mask] = foreground[binary_mask]
    img_mask[countours, :] = 0
    return img_mask.astype(img.dtype)

def process_video(input_video, output_dir, sam_gap=5, max_obj_num=255, min_area=200):
    """Process video with tracking and segmentation"""
    
    print(f"Processing video: {input_video}")
    print(f"Output directory: {output_dir}")
    
    # Create output directories
    output_mask_dir = os.path.join(output_dir, 'masks')
    os.makedirs(output_mask_dir, exist_ok=True)
    
    # Configure SAM parameters for good initialization
    sam_args['generator_args'] = {
        'points_per_side': 30,
        'pred_iou_thresh': 0.8,
        'stability_score_thresh': 0.9,
        'crop_n_layers': 1,
        'crop_n_points_downscale_factor': 2,
        'min_mask_region_area': min_area,
    }
    
    # Configure segtracker parameters
    segtracker_args_local = {
        'sam_gap': sam_gap,
        'min_area': min_area,
        'max_obj_num': max_obj_num,
        'min_new_obj_iou': 0.8,
    }
    
    # Initialize video capture
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video info: {total_frames} frames at {fps} FPS")
    
    # Initialize tracker
    segtracker = SegTracker(segtracker_args_local, sam_args, aot_args)
    segtracker.restart_tracker()
    
    pred_list = []
    frame_idx = 0
    
    print("Starting video processing...")
    
    with torch.cuda.amp.autocast():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if frame_idx == 0:
                print("Processing first frame - running SAM segmentation...")
                pred_mask = segtracker.seg(frame)
                torch.cuda.empty_cache()
                gc.collect()
                segtracker.add_reference(frame, pred_mask)
                print(f"Found {len(np.unique(pred_mask)) - 1} objects in first frame")
                
            elif (frame_idx % sam_gap) == 0:
                print(f"Frame {frame_idx}: Running SAM + tracking...")
                seg_mask = segtracker.seg(frame)
                torch.cuda.empty_cache()
                gc.collect()
                track_mask = segtracker.track(frame)
                
                # Find new objects and update tracker
                new_obj_mask = segtracker.find_new_objs(track_mask, seg_mask)
                save_prediction(new_obj_mask, output_mask_dir, f'{frame_idx}_new.png')
                pred_mask = track_mask + new_obj_mask
                segtracker.add_reference(frame, pred_mask)
                
            else:
                pred_mask = segtracker.track(frame, update_memory=True)
            
            torch.cuda.empty_cache()
            gc.collect()
            
            # Save mask
            save_prediction(pred_mask, output_mask_dir, f'{frame_idx}.png')
            pred_list.append(pred_mask)
            
            print(f"Processed frame {frame_idx}/{total_frames} - {segtracker.get_obj_num()} objects", end='\r')
            frame_idx += 1
    
    cap.release()
    print(f"\nFinished processing {frame_idx} frames")
    
    # Save visualization video
    output_video = os.path.join(output_dir, 'output_video.mp4')
    save_visualization_video(input_video, pred_list, output_video)
    
    # Save GIF
    output_gif = os.path.join(output_dir, 'output_masks.gif')
    imageio.mimsave(output_gif, pred_list, fps=fps)
    print(f"Saved visualization GIF: {output_gif}")
    
    # Cleanup
    del segtracker
    torch.cuda.empty_cache()
    gc.collect()
    
    return output_video, len(pred_list)

def save_visualization_video(input_video, pred_list, output_video):
    """Create visualization video with mask overlays"""
    print("Creating visualization video...")
    
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frame_idx = 0
    while cap.isOpened() and frame_idx < len(pred_list):
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred_mask = pred_list[frame_idx]
        masked_frame = draw_mask(frame, pred_mask)
        masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
        out.write(masked_frame)
        
        print(f'Writing frame {frame_idx}', end='\r')
        frame_idx += 1
    
    out.release()
    cap.release()
    print(f"\nSaved visualization video: {output_video}")

def main():
    parser = argparse.ArgumentParser(description="CLI Video Tracking and Segmentation")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("-o", "--output", default="./output", help="Output directory (default: ./output)")
    parser.add_argument("--sam-gap", type=int, default=5, help="Interval to run SAM (default: 5)")
    parser.add_argument("--max-objects", type=int, default=255, help="Maximum objects to track (default: 255)")
    parser.add_argument("--min-area", type=int, default=200, help="Minimum mask area (default: 200)")
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
    
    print("=" * 50)
    print("CLI Video Tracking and Segmentation")
    print("=" * 50)
    print(f"Input video: {args.input_video}")
    print(f"Output directory: {args.output}")
    print(f"SAM gap: {args.sam_gap}")
    print(f"Max objects: {args.max_objects}")
    print(f"Min area: {args.min_area}")
    print(f"Device: {args.device}")
    print("=" * 50)
    
    try:
        output_video, num_frames = process_video(
            args.input_video, 
            args.output,
            sam_gap=args.sam_gap,
            max_obj_num=args.max_objects,
            min_area=args.min_area
        )
        
        print("\n" + "=" * 50)
        print("PROCESSING COMPLETE!")
        print("=" * 50)
        print(f"Processed {num_frames} frames")
        print(f"Output video: {output_video}")
        print(f"Masks saved in: {os.path.join(args.output, 'masks')}")
        print("=" * 50)
        
        return 0
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())