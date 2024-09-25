def xyxy2xywh(x):
    import torch
    import numpy as np

    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y

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

    return dataset_path

def process_annotations(dataset_path, annotation_file):
    import os
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