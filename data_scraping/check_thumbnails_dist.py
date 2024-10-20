# %%
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import json
import torch
import pandas as pd
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from dataclasses import dataclass
from typing import List


# Define paths
base_dir = Path("/Users/vigji/My Drive/eco-beauty")
beautiful_dir = base_dir / "dataset/raw/beautiful"
ugly_dir = base_dir / "dataset/raw/ugly"
thumbnails_dir = base_dir / "google_lens_search/thumbnails"
annotations_dir = base_dir / "google_lens_search/annotations"

figures_dir = base_dir / "google_lens_search" / "composite_images"
figures_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ThumbnailData:
    image: Image
    thumbnail_number_id: int
    label: str
    score: float = None


@dataclass
class FocalImageData:
    image: Image
    thumbnails: List[ThumbnailData]
    category_id: str
    img_number_id: int


def load_image(category, img_number):
    # add trailing zeros to img_number if necessary:
    file_path = list(base_dir.glob(f"dataset/raw/{category}/img{img_number:03}.png"))[0]
    assert file_path.exists(), f"Image file not found: {file_path}"
    return Image.open(file_path)


def load_thumbnails(category, img_number, thumbnail_number_id):
    file_path = (
        thumbnails_dir / f"results_{category}_{img_number}_{thumbnail_number_id:03}.png"
    )
    assert file_path.exists(), f"Thumbnail file not found: {file_path}"
    return Image.open(file_path)


def load_annotations(category_id, img_number_id):
    """
    Load annotations for a given focal image.
    """
    annotation_file = (
        annotations_dir / f"{category_id}_{img_number_id:03}_annotations.json"
    )
    assert annotation_file.exists(), f"Annotation file not found: {annotation_file}"
    if annotation_file.exists():
        with open(annotation_file, "r") as f:
            return json.load(f)
    return None


def load_focal_image_data(category_id: str, img_number_id: str):
    image = load_image(category_id, img_number_id)
    annotations = load_annotations(category_id, img_number_id)

    thumbnails = []
    # Sadly we used bad sorting for the file system to name thumbnails in annotations, not
    # the right thumbnail_number_id.
    all_thumbnail_files = list(
        thumbnails_dir.glob(f"results_{category_id}_{img_number_id}_*.png")
    )
    if len(all_thumbnail_files) == 0:
        print(f"No thumbnails found for {category_id}_{img_number_id}")
        return None

    # print(all_thumbnail_files)
    for label, thumbnail_file in zip(annotations.values(), all_thumbnail_files):
        thumbnail_number_id = int(thumbnail_file.stem.split("_")[-1])
        thumbnail_image = load_thumbnails(
            category_id, img_number_id, thumbnail_number_id
        )
        thumbnail_data = ThumbnailData(thumbnail_image, thumbnail_number_id, label)
        thumbnails.append(thumbnail_data)

    return FocalImageData(image, thumbnails, category_id, img_number_id)


def load_all_focal_image_data():
    all_focal_image_data = []
    for category_id in ["beautiful", "ugly"]:
        annotation_files = list(annotations_dir.glob(f"{category_id}_*.json"))
        for annotation_file in tqdm(annotation_files):
            # print(annotation_file.stem)
            category_id, img_number_id = annotation_file.stem.split("_")[:2]
            focal_image_data = load_focal_image_data(category_id, img_number_id)
            if focal_image_data is not None:
                all_focal_image_data.append(focal_image_data)
    return all_focal_image_data


def setup_image_processing():
    """
    Set up the image processing pipeline.
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


def compute_similarity_score(image_data: FocalImageData, feature_extractor, transform):

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    feature_extractor.to(device)
    feature_extractor.eval()

    focal_tensor = transform(image_data.image).unsqueeze(0).to(device)
    with torch.no_grad():
        focal_features = feature_extractor(focal_tensor).squeeze(0)

    thumbnail_tensors = torch.stack(
        [transform(thumbnail.image) for thumbnail in image_data.thumbnails]
    ).to(device)

    with torch.no_grad():
        thumbnail_features = feature_extractor(thumbnail_tensors)

    similarities = cosine_similarity(focal_features.unsqueeze(0), thumbnail_features)
    similarities = similarities.cpu().numpy().flatten()

    for thumbnail, similarity in zip(image_data.thumbnails, similarities):
        thumbnail.score = similarity
    return similarities


def create_similarity_dataframe(all_imgs):
    data = []
    for img in tqdm(all_imgs):
        for thumbnail in img.thumbnails:
            data.append(
                {
                    "focal_image_id": img.img_number_id,
                    "similarity": thumbnail.score,
                    "true_label": thumbnail.label,
                    "thumbnail_id": thumbnail.thumbnail_number_id,
                }
            )

    return pd.DataFrame(data)


def create_composite_image(focal_image_data: FocalImageData, transform_visualization):
    """
    Create a composite image of thumbnails with similarity scores.
    """
    num_columns = 10
    num_rows = (len(focal_image_data.thumbnails) + num_columns - 1) // num_columns
    composite_image = Image.new("RGB", (256 * num_columns, 256 * num_rows))

    thumbnails_with_scores = []
    for thumbnail_data in focal_image_data.thumbnails:
        thumbnail_copy = thumbnail_data.image.copy()
        thumbnail_copy = transform_visualization(thumbnail_copy)
        thumbnail_copy = transforms.ToPILImage()(thumbnail_copy.squeeze(0).clamp(0, 1))

        color = "green" if thumbnail_data.label == "same" else "red"
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
        draw.text((10, 10), f"{thumbnail_data.score:.4f}", fill="red")
        thumbnails_with_scores.append((thumbnail_data.score, framed_thumbnail))

    sorted_thumbnails_with_scores = [
        x
        for _, x in sorted(
            thumbnails_with_scores, key=lambda pair: pair[0], reverse=True
        )
    ]

    for i, thumbnail in enumerate(sorted_thumbnails_with_scores):
        row, col = divmod(i, num_columns)
        composite_image.paste(thumbnail, (col * 256, row * 256))

    return composite_image


composite_image = create_composite_image(all_imgs[0], transform_visualization)
image = load_focal_image_data("beautiful", "025")
all_imgs = load_all_focal_image_data()


feature_extractor, transform, transform_visualization = setup_image_processing()

for img in all_imgs:
    compute_similarity_score(img, feature_extractor, transform)
    composite_image = create_composite_image(img, transform_visualization)
    composite_image.save(
        figures_dir / f"{img.category_id}_{img.img_number_id}_composite.png"
    )

# %%
# Some analysis plots

df = create_similarity_dataframe(all_imgs)
# %%
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.swarmplot(data=df, x="true_label", y="similarity", color="blue")
plt.show()

# Plot proportion of false positives for different numbers of most similar thumbnails
n_thumbnails = np.arange(3, 30, 1)
false_positive_rates = []

for n in n_thumbnails:
    # Sort thumbnails by similarity for each focal image
    df_sorted = df.sort_values(
        ["focal_image_id", "similarity"], ascending=[True, False]
    )

    # Select top n thumbnails for each focal image
    df_top_n = df_sorted.groupby("focal_image_id").head(n)

    # Calculate false positive rate
    false_positives = df_top_n[df_top_n["true_label"] != "same"].shape[0]
    total = df_top_n.shape[0]
    false_positive_rate = false_positives / total
    false_positive_rates.append(false_positive_rate)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(n_thumbnails, false_positive_rates, marker="o")
plt.xlabel("Number of most similar thumbnails")
plt.ylabel("Proportion of false positives")
plt.title("False Positive Rate vs Number of Most Similar Thumbnails")
plt.xticks(n_thumbnails)
plt.ylim(0, 1)
plt.grid(True)
plt.show()
