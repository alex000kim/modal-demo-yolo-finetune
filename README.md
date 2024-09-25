# YOLOv10 SKU-110K Object Detection

This project implements object detection on the SKU-110K dataset using YOLOv10. It uses Modal for distributed computing and GPU acceleration.

## Features

- Downloads and processes the SKU-110K dataset
- Trains a YOLOv10 model on the dataset
- Creates preview images with bounding box predictions

## Requirements

- Modal
- Python 3.10

## Setup

1. Install Modal: `pip install modal`
2. Run `modal setup` to authenticate (if this doesn’t work, try `python -m modal setup`)

## Usage
- You might want to update the volume name in the script to your own volume name.
-Run the main script:

```bash
modal run yolo-sku110k.py
```

This will download the dataset, train the model, and create preview images.

