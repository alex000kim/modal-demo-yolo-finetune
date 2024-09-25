import modal
from pathlib import Path
from helpers import download_and_extract_dataset, process_annotations, create_sample_file

app = modal.App("yolov9-sku110k")

cpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "pandas",
        "numpy",
        "tqdm",
        "torch",
        "ultralytics",
    )
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0"])
    .pip_install(
        "torch",
        "ultralytics",
    )
)

vol = modal.Volume.from_name("alexkim-volume", create_if_missing=True)

@app.function(image=cpu_image, 
              volumes={"/my-mount": vol}, 
              cpu=1.0,
              timeout=60*60 # 1 hour
              )
def download_and_process_dataset():
    dataset_path = Path(download_and_extract_dataset())
    for annotation_file in ['annotations_train.csv', 'annotations_val.csv', 'annotations_test.csv']:
        process_annotations(dataset_path, annotation_file)   
    for file in ['train.txt', 'test.txt', 'val.txt']:
        create_sample_file(dataset_path / file)
    vol.commit()
    return dataset_path

@app.function(image=gpu_image, 
              gpu="A10G", 
              volumes={"/my-mount": vol},
              timeout=60*60*24 # 1 day
              )
def train(dataset_path):
    import shutil
    from ultralytics import YOLO

    def create_dataset_config(dataset_path: Path) -> str:
        return f"""
path: {dataset_path}
train: sample_train.txt
val: sample_val.txt
test: sample_test.txt

nc: 1
names: ['object']
        """

    dataset_config = create_dataset_config(dataset_path)
    dataset_config_path = dataset_path / "SKU-110K.yaml"
    dataset_config_path.write_text(dataset_config)

    print("Initializing YOLOv10 model...")
    model = YOLO('yolov10n.yaml') 

    args = dict(
        data=dataset_config_path,
        epochs=10,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        name='yolov10-sku110k'
    )

    print("Starting training...")
    results = model.train(**args)
    best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
    best_model_save_path = dataset_path / "yolov10-sku110k-best.pt"
    # copy to volume
    shutil.copy(best_model_path, best_model_save_path)
    print(f"Best model saved to {best_model_save_path}")

    vol.commit()
    return best_model_save_path

@app.function(image=gpu_image, 
              gpu="A10G", 
              volumes={"/my-mount": vol},
              timeout=60*60 # 1 hour
              )
def create_preview_images(dataset_path, model_path, num_images=10):
    import cv2
    import matplotlib.pyplot as plt
    from ultralytics import YOLO
    import glob

    def process_image(img_path: str, model: YOLO, preview_dir: Path) -> None:
        img = cv2.imread(img_path)
        results = model(img)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
        
        output_path = preview_dir / f"preview_{Path(img_path).name}"
        plt.savefig(output_path)
        plt.close(fig)

    model = YOLO(model_path)
    images_f_list = glob.glob(str(dataset_path/"images"/"*.jpg"))
    preview_dir = dataset_path / "preview_images"
    preview_dir.mkdir(exist_ok=True)

    for img_path in images_f_list[:num_images]:
        process_image(img_path, model, preview_dir)

    print(f"Preview images saved to {preview_dir}")
    vol.commit()

@app.local_entrypoint()
def main():
    dataset_path = download_and_process_dataset.remote()
    best_model_save_path = train.remote(dataset_path)
    create_preview_images.remote(dataset_path, best_model_save_path)