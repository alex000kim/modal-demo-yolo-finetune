import os
import modal


app = modal.App("yolov9-sku110k")


cpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "pandas",
        "numpy",
        "Pillow",
        "tqdm",
        "matplotlib",
        "torch",
        "torchvision",
        "ultralytics",
        "onnx",
        "onnxruntime-gpu",
    )
)


gpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0"])
    .pip_install(
        "torch",
        "torchvision",
        "ultralytics",
        "onnx",
        "onnxruntime-gpu",
    )
)

vol = modal.Volume.from_name("alexkim-volume", create_if_missing=True)


# Helper functions
def xyxy2xywh(x):
    import torch
    import numpy as np

    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


@app.function(image=cpu_image, 
              volumes={"/my-mount": vol}, 
              cpu=1.0,
              timeout=60*60 # 1 hour
              )
def create_sample_file(input_file, sample_size_percentage=10):
    import os
    import random

    dir_name, file_name = os.path.split(input_file)
    name_part, extension = os.path.splitext(file_name)
    sample_file_name = f'sample_{name_part}{extension}'
    sample_file_path = os.path.join(dir_name, sample_file_name)

    if os.path.exists(sample_file_path):
        print(f'Sample file already exists: {sample_file_path}')
    else:
        with open(input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        lines_to_select = max(1, len(lines) * sample_size_percentage // 100)
        sampled_lines = random.sample(lines, lines_to_select)
        with open(sample_file_path, 'w', encoding='utf-8') as file:
            file.writelines(sampled_lines)

        print(f'Sample file created: {sample_file_path}')
    vol.commit()

@app.function(image=cpu_image, 
              volumes={"/my-mount": vol}, 
              cpu=1.0,
              timeout=60*60 # 1 hour
              )
def download_and_extract_dataset():
    import os
    from pathlib import Path
    import tarfile
    import urllib.request
    
    dataset_url = "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"
    dataset_path = Path("/my-mount/SKU110K_fixed")
    tar_file_path = dataset_path.with_suffix('.tar.gz')

    if not dataset_path.exists():
        print("Downloading dataset...")
        urllib.request.urlretrieve(dataset_url, tar_file_path)
        
        print("Extracting dataset...")
        with tarfile.open(tar_file_path, 'r:gz') as tar:
            tar.extractall(path=dataset_path.parent)    
        os.remove(tar_file_path)
        (dataset_path / 'labels').mkdir(parents=True, exist_ok=True)
    else:
        print("Dataset already exists.")

    vol.commit()
    return dataset_path

@app.function(image=cpu_image, 
              volumes={"/my-mount": vol}, 
              cpu=1.0,
              timeout=60*60 # 1 hour
              )
def process_annotations(dataset_path, annotation_file):
    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    out_file = (dataset_path / annotation_file).with_suffix('.txt').__str__().replace('annotations_', '')
    if os.path.exists(out_file):
        print(f'{out_file} already exists.')
    else:
        names = 'image', 'x1', 'y1', 'x2', 'y2', 'class', 'image_width', 'image_height'
        x = pd.read_csv(dataset_path / 'annotations' / annotation_file, names=names).values
        images, unique_images = x[:, 0], np.unique(x[:, 0])
        
        for im in tqdm(unique_images, desc=f'Converting {dataset_path / "annotations" / annotation_file}'):
            cls = 0
            with open((dataset_path / 'labels' / im).with_suffix('.txt'), 'w') as f:
                for r in x[images == im]:
                    w, h = r[6], r[7]
                    xywh = xyxy2xywh(np.array([[r[1] / w, r[2] / h, r[3] / w, r[4] / h]]))[0]
                    f.write(f"{cls} {xywh[0]:.5f} {xywh[1]:.5f} {xywh[2]:.5f} {xywh[3]:.5f}\n")
        print(f'Writing {out_file}')
        with open(out_file, 'w') as f:
            f.writelines(f'./images/{s}\n' for s in unique_images)
    vol.commit()


@app.function(image=gpu_image, 
              gpu="A10G", 
              volumes={"/my-mount": vol},
              timeout=60*60*24 # 1 day
              )
def train(dataset_path):
    import os
    from pathlib import Path
    from ultralytics import YOLO

    dataset_config = f"""
path: {dataset_path}
train: sample_train.txt
val: sample_val.txt
test: sample_test.txt

nc: 1
names: ['object']
    """
    dataset_config_path = dataset_path/"SKU-110K.yaml"
    with open(dataset_config_path, "w") as f:
        f.write(dataset_config)

    # Initialize YOLOv10 model
    print("Initializing YOLOv10 model...")
    model = YOLO('yolov10n.yaml') 

    # Set up training arguments
    args = dict(
        data=dataset_config_path,
        epochs=10,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        name='yolov10-sku110k'
    )

    # Run training
    print("Starting training...")
    results = model.train(**args)

    # Export to ONNX
    print("Exporting to ONNX...")
    model.export(format='onnx')

    print("Training and export completed.")
    
    # Save the trained model to the volume
    best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
    save_path = dataset_path/"yolov10-sku110k-best.pt"
    os.system(f"cp {best_model_path} {save_path}")
    print(f"Best model saved to {save_path}")

    vol.commit()



@app.local_entrypoint()
def main():
    dataset_path = download_and_extract_dataset.remote()
    for annotation_file in ['annotations_train.csv', 'annotations_val.csv', 'annotations_test.csv']:
        process_annotations.remote(dataset_path, annotation_file)

    for file in ['train.txt', 'test.txt', 'val.txt']:
        create_sample_file.remote(dataset_path / file)
    train.remote(dataset_path)