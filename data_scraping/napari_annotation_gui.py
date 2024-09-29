import napari
import numpy as np
from pathlib import Path
from PIL import Image
import json

# Define paths
base_dir = Path("/Users/vigji/My Drive/eco-beauty")
beautiful_dir = base_dir / "dataset/raw/beautiful"
ugly_dir = base_dir / "dataset/raw/ugly"
thumbnails_dir = base_dir / "google_lens_search/thumbnails"
annotations_dir = base_dir / "google_lens_search/annotations"
annotations_dir.mkdir(parents=True, exist_ok=True)

def load_image(category, img_number):
    file_path = list(base_dir.glob(f"dataset/raw/{category}/img{img_number:03}.png"))[0]
    return np.array(Image.open(file_path))

def load_thumbnails(category, img_number):
    file_paths = list(thumbnails_dir.glob(f"results_{category}_{img_number:03}_*.png"))
    return [np.array(Image.open(file_path)) for file_path in file_paths]

def annotate_images(focal_image_id):
    category, img_number = focal_image_id.split("_")[0], int(focal_image_id.split("_")[1])
    focal_image = load_image(category, img_number)
    thumbnails = load_thumbnails(category, img_number)

    viewer = napari.Viewer()
    focal_layer = viewer.add_image(focal_image, name="Focal Image")

    annotations = {}

    for i, thumbnail in enumerate(thumbnails):
        thumbnail_layer = viewer.add_image(thumbnail, name=f"Thumbnail {i+1}")
        
        # Center the layers side by side
        focal_layer.translate = (-focal_image.shape[1]/2, 0)
        thumbnail_layer.translate = (thumbnail.shape[1]/2, 0)

        # Wait for keyboard input
        while True:
            key = input(f"Thumbnail {i+1}: Press 's' for same, 'd' for different, or 'q' to quit: ").lower()
            if key in ['s', 'd', 'q']:
                break
        
        if key == 'q':
            break
        
        annotations[f"thumbnail_{i+1}"] = "same" if key == 's' else "different"
        viewer.layers.remove(thumbnail_layer)

    viewer.close()

    # Save annotations
    annotation_file = annotations_dir / f"{focal_image_id}_annotations.json"
    with open(annotation_file, 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"Annotations saved to {annotation_file}")

if __name__ == "__main__":
    focal_image_id = input("Enter the focal image ID (e.g., 'beautiful_000'): ")
    annotate_images(focal_image_id)