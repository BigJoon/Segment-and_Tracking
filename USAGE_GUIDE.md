# Single Object Tracking - Usage Guide

## Overview
The single object tracking system allows you to track one specific object through a video by selecting it with either a bounding box or point click. The output is a white mask video showing only the selected object.

## Setup
```bash
# Activate environment
conda activate sam-track

# Navigate to project directory
cd /work/kt_bigjoon/Segment-and-Track-Anything
```

## Usage

### 1. Point Selection (Click on Object)
```bash
python single_object_tracker.py input_video.mp4 --point X Y [options]
```

**Example:**
```bash
python single_object_tracker.py assets/cell.mp4 --point 640 360 -o ./output
```

### 2. Bounding Box Selection
```bash
python single_object_tracker.py input_video.mp4 --bbox X1 Y1 X2 Y2 [options]
```

**Example:**
```bash
python single_object_tracker.py assets/cell.mp4 --bbox 600 320 680 400 -o ./output
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o OUTPUT` | Output directory | `./single_track_output` |
| `--mask-only` | Output only white mask video (no overlay) | False |
| `--device DEVICE` | Device to use (cuda/cpu) | `cuda` |

## Output Files

### Standard Output (both selection methods):
- `mask_video.mp4` - **White mask video** (object = white, background = black)
- `masks/` - Individual frame masks as PNG files (000000.png, 000001.png, ...)
- `overlay_video.mp4` - Original video with colored overlay (unless `--mask-only`)

### File Descriptions:
- **mask_video.mp4**: Pure white mask video showing only the tracked object
- **overlay_video.mp4**: Original video with green overlay showing the tracked object
- **masks/*.png**: Individual grayscale mask images (white = object, black = background)

## Examples

### Example 1: Track cell with point selection
```bash
python single_object_tracker.py assets/cell.mp4 --point 640 360 --mask-only -o cell_tracking
```

### Example 2: Track object with bounding box
```bash
python single_object_tracker.py assets/cell.mp4 --bbox 500 200 700 400 -o bbox_tracking
```

### Example 3: Track with both mask and overlay output
```bash
python single_object_tracker.py assets/cell.mp4 --point 400 300 -o full_output
```

## Tips for Selection

### Point Selection (`--point X Y`):
- Click on the center of the object you want to track
- Works well for distinct objects with clear boundaries
- Single point defines the object of interest

### Bounding Box Selection (`--bbox X1 Y1 X2 Y2`):
- Define a rectangle around the object: (X1,Y1) = top-left, (X2,Y2) = bottom-right
- More precise for selecting specific parts of objects
- Better for complex scenes with multiple similar objects

### Finding Good Coordinates:
1. Use any image viewer to check the first frame coordinates
2. Video dimensions for cell.mp4: 1280 x 720
3. Center point: (640, 360)

## Output Quality

### Successful Tracking Indicators:
- Object area > 0 pixels in selection step
- Consistent tracking through frames
- Clean white mask in output video

### Troubleshooting:
- If "No object found": Try different coordinates
- If tracking fails: Object might be too small or unclear
- If multiple objects: Use smaller bounding box

## Integration with Your Code

The core functions can be imported:

```python
from single_object_tracker import get_first_frame_selection, track_single_object

# Get object mask from first frame
first_frame, object_mask = get_first_frame_selection(
    video_path, "point", [x, y]  # or "bbox", [x1, y1, x2, y2]
)

# Track through video
mask_list, num_frames = track_single_object(
    video_path, first_frame, object_mask, output_dir
)
```

## Performance

- **Processing Speed**: ~2-3 FPS depending on video resolution
- **Memory Usage**: ~2-4GB GPU memory for 720p video
- **Output Size**: Mask video ~1-2MB, Overlay video ~20-30MB for typical cell video