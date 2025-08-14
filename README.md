# Segment-and-Tracking

A CLI-based video object segmentation and tracking tool built on **Segment Anything Model (SAM)** and **Associating Objects with Transformers (AOT)**. This project provides both automatic multi-object tracking and precise single-object tracking with interactive selection.

## Features

### 🎯 Single Object Tracking
- **Interactive Selection**: Choose objects using point clicks or bounding boxes
- **White Mask Output**: Generate clean white mask videos showing only the selected object
- **High Precision**: Track specific objects throughout entire videos
- **Flexible Output**: Both mask-only and overlay visualization options

### 🔄 Multi-Object Tracking  
- **Automatic Detection**: SAM-powered automatic object detection
- **Comprehensive Tracking**: Track all detected objects simultaneously
- **Configurable Parameters**: Adjust detection sensitivity and tracking parameters

## Installation

### Prerequisites
- CUDA-capable GPU (recommended)
- Anaconda/Miniconda
- Python 3.9

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/BigJoon/Segment-and_Tracking.git
cd Segment-and_Tracking

# Create conda environment
conda create -n sam-track python=3.9 -y
conda activate sam-track

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio cudatoolkit=11.3 -c pytorch -y

# Install dependencies
pip install opencv-python pillow numpy imageio gdown

# Install SAM
pip install -e ./sam

# Download model checkpoints
mkdir -p ckpt
cd ckpt
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
gdown '1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ' --output R50_DeAOTL_PRE_YTB_DAV.pth
cd ..

# Setup GroundingDINO (for text detection features)
git clone https://github.com/IDEA-Research/GroundingDINO.git
cp -r GroundingDINO/groundingdino .
```

## Quick Start

### Single Object Tracking

Track a specific object by pointing to it:
```bash
python single_object_tracker.py video.mp4 --point 640 360 --mask-only -o output
```

Track a specific object with bounding box:
```bash
python single_object_tracker.py video.mp4 --bbox 500 200 700 400 -o output
```

### Multi-Object Tracking

Track all objects automatically:
```bash
python cli_track.py video.mp4 -o output --sam-gap 10 --max-objects 50
```

## Usage Examples

### Single Object Tracking Options
```bash
# Point selection (click coordinates)
python single_object_tracker.py input.mp4 --point X Y [options]

# Bounding box selection (rectangle coordinates)  
python single_object_tracker.py input.mp4 --bbox X1 Y1 X2 Y2 [options]

Options:
  -o OUTPUT         Output directory (default: ./single_track_output)
  --mask-only       Output only white mask video (no overlay)
  --device DEVICE   Device to use: cuda/cpu (default: cuda)
```

### Multi-Object Tracking Options
```bash
python cli_track.py input.mp4 [options]

Options:
  -o OUTPUT           Output directory (default: ./output)
  --sam-gap N         Interval to run SAM (default: 5)
  --max-objects N     Maximum objects to track (default: 255)
  --min-area N        Minimum mask area (default: 200)
  --device DEVICE     Device to use (default: cuda)
```

## Output

### Single Object Tracking
- `mask_video.mp4` - **White mask video** (main output)
- `overlay_video.mp4` - Original video with tracking overlay
- `masks/` - Individual frame masks as PNG files

### Multi-Object Tracking
- `output_video.mp4` - Video with all tracked objects
- `output_masks.gif` - Animated mask sequence
- `masks/` - Individual frame masks with object IDs

## Sample Results

The tool has been tested on various video types:
- **Cell microscopy videos**: Precise cell tracking and division detection
- **Object motion videos**: Robust tracking through occlusions
- **Multi-object scenes**: Simultaneous tracking of multiple targets

## Technical Details

### Core Components
- **SAM Integration**: Automatic mask generation and interactive segmentation
- **AOT Tracking**: Transformer-based object association across frames
- **Memory Management**: Efficient processing of long video sequences
- **CLI Interface**: User-friendly command-line tools

### Model Specifications
- **SAM Model**: ViT-B (358MB) - Vision Transformer backbone
- **AOT Model**: R50-DeAOTL (237MB) - ResNet-50 with DeAOT layers
- **Input Resolution**: Supports various video resolutions
- **Performance**: ~2-3 FPS processing speed on modern GPUs

## Troubleshooting

### Common Issues
- **CUDA out of memory**: Reduce video resolution or batch size
- **No object found**: Adjust selection coordinates or area thresholds
- **Tracking drift**: Fine-tune SAM gap and IoU thresholds

### Performance Tips
- Use `--sam-gap` to balance accuracy vs speed
- Adjust `--min-area` to filter small objects
- Use `--mask-only` for faster processing

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project builds upon several open-source projects:
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) - Meta AI
- [AOT (Associating Objects with Transformers)](https://github.com/yoxu515/aot-benchmark)
- [Segment-and-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything) - Original implementation

## Acknowledgments

Special thanks to the original authors of Segment-and-Track-Anything and the teams behind SAM and AOT for their groundbreaking work in computer vision and object tracking.

---

## Citation

If you use this work in your research, please cite the original papers:

```bibtex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

@article{yang2023aot,
  title={Associating Objects with Transformers for Video Object Segmentation},
  author={Yang, Zongxin and Wei, Yunchao and Yang, Yi},
  journal={Advances in Neural Information Processing Systems},
  year={2021}
}
```