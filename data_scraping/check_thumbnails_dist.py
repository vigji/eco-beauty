# %%
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import json
import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.functional import cosine_similarity


# Define paths
base_dir = Path("/Users/vigji/My Drive/eco-beauty")
beautiful_dir = base_dir / "dataset/raw/beautiful"
ugly_dir = base_dir / "dataset/raw/ugly"
thumbnails_dir = base_dir / "google_lens_search/thumbnails"
annotations_dir = base_dir / "google_lens_search/annotations"


def load_image(category, img_number):
    # add trailing zeros to img_number if necessary:
    file_path = list(base_dir.glob(f"dataset/raw/{category}/img{img_number:03}.png"))[0]
    return Image.open(file_path)


def load_thumbnails(category, img_number):
    file_paths = list(thumbnails_dir.glob(f"results_{category}_{img_number:03}_*.png"))
    return [Image.open(file_path) for file_path in file_paths]


def load_annotations(annotations_dir, focal_image_id):
    """
    Load annotations for a given focal image.

    Parameters
    ----------
    annotations_dir : Path
        Directory containing annotation files.
    focal_image_id : str
        ID of the focal image.

    Returns
    -------
    dict
        Annotations for the focal image.
    """
    annotation_file = annotations_dir / f"{focal_image_id}_annotations.json"
    if annotation_file.exists():
        with open(annotation_file, "r") as f:
            return json.load(f)
    return None


def setup_image_processing():
    """
    Set up the image processing pipeline.

    Returns
    -------
    tuple
        Contains the feature extractor, transform, and transform_visualization.
    """
    resnet = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V1
    )  # or DEFAULT
    resnet.eval()
    layers_to_keep = list(resnet.children())[:-1]
    feature_extractor = torch.nn.Sequential(*layers_to_keep)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transform_visualization = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    return feature_extractor, transform, transform_visualization


def extract_features(image, feature_extractor, transform):
    """
    Extract features from an image.

    Parameters
    ----------
    image : PIL.Image
        Input image.
    feature_extractor : torch.nn.Module
        Feature extraction model.
    transform : torchvision.transforms.Compose
        Image transformation pipeline.

    Returns
    -------
    torch.Tensor
        Extracted features.
    """
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(img_tensor)
    return features.squeeze()


def create_composite_image(
    focal_image, focal_thumbnails, similarities, annotations, transform_visualization
):
    """
    Create a composite image of thumbnails with similarity scores.

    Parameters
    ----------
    focal_image : PIL.Image
        The focal image.
    focal_thumbnails : list
        List of thumbnail images.
    similarities : list
        List of similarity scores.
    annotations : dict
        Annotations for the thumbnails.
    transform_visualization : torchvision.transforms.Compose
        Image transformation pipeline for visualization.

    Returns
    -------
    PIL.Image
        Composite image of thumbnails with similarity scores.
    """
    num_columns = 10
    num_rows = (len(focal_thumbnails) + num_columns - 1) // num_columns
    composite_image = Image.new("RGB", (256 * num_columns, 256 * num_rows))

    thumbnails_with_scores = []
    for i, (thumbnail, label) in enumerate(zip(focal_thumbnails, annotations.values())):
        thumbnail_copy = thumbnail.copy()
        thumbnail_copy = transform_visualization(thumbnail_copy)
        thumbnail_copy = transforms.ToPILImage()(thumbnail_copy.squeeze(0).clamp(0, 1))

        color = "green" if label == "same" else "red"
        framed_thumbnail = Image.new("RGB", thumbnail_copy.size, color=color)
        width_pxs = 4
        cropped_thumbnail = thumbnail_copy.crop(
            (
                0,
                0,
                framed_thumbnail.width - 2 * width_pxs,
                framed_thumbnail.height - 2 * width_pxs,
            )
        )
        framed_thumbnail.paste(cropped_thumbnail, (width_pxs, width_pxs))

        draw = ImageDraw.Draw(framed_thumbnail)
        draw.text((10, 10), f"{similarities[i]:.4f}", fill="red")
        thumbnails_with_scores.append(framed_thumbnail)

    sorted_thumbnails_with_scores = [
        x
        for _, x in sorted(
            zip(similarities, thumbnails_with_scores),
            key=lambda pair: pair[0],
            reverse=True,
        )
    ]

    for i, thumbnail in enumerate(sorted_thumbnails_with_scores):
        row, col = divmod(i, num_columns)
        composite_image.paste(thumbnail, (col * 256, row * 256))

    return composite_image


def process_focal_image(
    focal_image_id, base_dir, feature_extractor, transform, transform_visualization
):
    """
    Process a single focal image and its thumbnails.

    Parameters
    ----------
    focal_image_id : str
        ID of the focal image.
    base_dir : Path
        Base directory for the project.
    feature_extractor : torch.nn.Module
        Feature extraction model.
    transform : torchvision.transforms.Compose
        Image transformation pipeline.
    transform_visualization : torchvision.transforms.Compose
        Image transformation pipeline for visualization.

    Returns
    -------
    PIL.Image
        Composite image of thumbnails with similarity scores.
    """
    category, img_number = focal_image_id.split("_")[0], int(
        focal_image_id.split("_")[1]
    )
    focal_image = load_image(category, img_number)
    focal_thumbnails = load_thumbnails(category, img_number)
    annotations = load_annotations(
        base_dir / "google_lens_search/annotations", focal_image_id
    )
    # print(annotations)

    if annotations is None or len(annotations) == 0:
        print(f"No annotations found for {focal_image_id}")
        return None

    focal_features = extract_features(focal_image, feature_extractor, transform)

    similarities = []
    for thumbnail in focal_thumbnails:
        thumbnail_features = extract_features(thumbnail, feature_extractor, transform)
        similarity = cosine_similarity(
            focal_features.unsqueeze(0), thumbnail_features.unsqueeze(0)
        )
        similarities.append(similarity.item())

    composite_image = create_composite_image(
        focal_image,
        focal_thumbnails,
        similarities,
        annotations,
        transform_visualization,
    )
    return composite_image


def main():
    base_dir = Path("/Users/vigji/My Drive/eco-beauty")
    annotations_dir = base_dir / "google_lens_search/annotations"
    composite_dir = base_dir / "google_lens_search/composite_images"
    composite_dir.mkdir(parents=True, exist_ok=True)

    feature_extractor, transform, transform_visualization = setup_image_processing()

    # Find all available annotation files
    annotation_files = list(annotations_dir.glob("*_annotations.json"))
    focal_image_ids = [
        file.stem.replace("_annotations", "") for file in annotation_files
    ]

    for focal_image_id in focal_image_ids:
        composite_image = process_focal_image(
            focal_image_id,
            base_dir,
            feature_extractor,
            transform,
            transform_visualization,
        )
        if composite_image:
            composite_image.save(composite_dir / f"{focal_image_id}_composite.png")
            print(f"Processed and saved composite image for {focal_image_id}")


if __name__ == "__main__":
    main()


# %%
