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
from tqdm import tqdm
from dataclasses import dataclass
from typing import List


# Define paths
base_dir = Path("/Users/vigji/My Drive/eco-beauty")
beautiful_dir = base_dir / "dataset/raw/beautiful"
ugly_dir = base_dir / "dataset/raw/ugly"
thumbnails_dir = base_dir / "google_lens_search/thumbnails"
annotations_dir = base_dir / "google_lens_search/annotations"


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
        thumbnails_dir
        / f"results_{category}_{img_number:03}_{thumbnail_number_id:03}.png"
    )
    assert file_path.exists(), f"Thumbnail file not found: {file_path}"
    return Image.open(file_path)


def load_annotations(annotations_dir, category_id, img_number_id):
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


def load_focal_image_data(base_dir, category_id, img_number_id):
    image = load_image(category_id, img_number_id)
    annotations = load_annotations(annotations_dir, category_id, img_number_id)

    thumbnails = []
    # Sadly we used bad sorting for the file system to name thumbnails in annotations, not
    # the right thumbnail_number_id.
    all_thumbnail_files = list(
        thumbnails_dir.glob(f"results_{category_id}_{img_number_id:03}_*.png")
    )

    print(all_thumbnail_files)
    for label, thumbnail_file in zip(annotations.values(), all_thumbnail_files):
        thumbnail_number_id = int(thumbnail_file.stem.split("_")[-1])
        thumbnail_image = load_thumbnails(
            category_id, img_number_id, thumbnail_number_id
        )
        thumbnail_data = ThumbnailData(thumbnail_image, thumbnail_number_id, label)
        thumbnails.append(thumbnail_data)

    return FocalImageData(image, thumbnails, category_id, img_number_id)


image = load_focal_image_data(base_dir, "beautiful", 25)
# %%

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

    similarities = cosine_similarity(
        focal_features.unsqueeze(0), thumbnail_features
    )
    similarities = similarities.cpu().numpy().flatten()

    for thumbnail, similarity in zip(image_data.thumbnails, similarities):
        thumbnail.score = similarity
    return similarities


feature_extractor, transform, transform_visualization = setup_image_processing()
compute_similarity_score(image, feature_extractor, transform)

# %%
image.thumbnails[0]
# %%

# TODO working here
def create_similarity_dataframe_gpu(base_dir, feature_extractor, transform):
    """
    Create a dataframe with similarity scores and true labels for all annotations using GPU acceleration.

    Parameters
    ----------
    base_dir : pathlib.Path
        The base directory containing the dataset and annotations.
    feature_extractor : torch.nn.Module
        The feature extractor model.
    transform : torchvision.transforms.Compose
        The transformation pipeline for the images.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing focal image IDs, similarity scores, true labels, and thumbnail IDs.
    """
    annotations_dir = base_dir / "google_lens_search/annotations"
    annotation_files = list(annotations_dir.glob("*_annotations.json"))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    feature_extractor.to(device)
    feature_extractor.eval()

    data = []
    for annotation_file in tqdm(annotation_files):
        focal_image_id = annotation_file.stem.replace("_annotations", "")
        annotations = load_annotations(annotations_dir, focal_image_id)
        if annotations is None or len(annotations) == 0:
            continue

        category, img_number = focal_image_id.split("_")[0], int(
            focal_image_id.split("_")[1]
        )
        focal_image = load_image(category, img_number)
        focal_thumbnails = load_thumbnails(category, img_number)
        if focal_image is None or not focal_thumbnails:
            continue

        focal_tensor = transform(focal_image).unsqueeze(0).to(device)
        with torch.no_grad():
            focal_features = feature_extractor(focal_tensor).squeeze(0)

        thumbnail_tensors = torch.stack(
            [transform(thumbnail) for thumbnail in focal_thumbnails]
        ).to(device)
        with torch.no_grad():
            thumbnail_features = feature_extractor(thumbnail_tensors)

        similarities = cosine_similarity(
            focal_features.unsqueeze(0), thumbnail_features
        )
        similarities = similarities.cpu().numpy().flatten()

        for (thumb_id, label), similarity in zip(annotations.items(), similarities):
            data.append(
                {
                    "focal_image_id": focal_image_id,
                    "similarity": similarity,
                    "true_label": label,
                    "thumbnail_id": thumb_id,
                }
            )

    return pd.DataFrame(data)
# %%


def extract_features(image, feature_extractor, transform):
    """
    Extract features from an image.
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


def process_all_images():
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


# if __name__ == "__main__":
#     main()

import pandas as pd


def create_similarity_dataframe(base_dir, feature_extractor, transform):
    """
    Create a dataframe with similarity scores and true labels for all annotations.
    """
    annotations_dir = base_dir / "google_lens_search/annotations"
    annotation_files = list(annotations_dir.glob("*_annotations.json"))

    feature_extractor, transform, _ = setup_image_processing()

    data = []
    for annotation_file in tqdm(annotation_files):
        focal_image_id = annotation_file.stem.replace("_annotations", "")
        annotations = load_annotations(annotations_dir, focal_image_id)
        if annotations is None or len(annotations) == 0:
            continue

        category, img_number = focal_image_id.split("_")[0], int(
            focal_image_id.split("_")[1]
        )
        focal_image = load_image(category, img_number)
        focal_thumbnails = load_thumbnails(category, img_number)
        if focal_image is None or not focal_thumbnails:
            continue

        focal_features = extract_features(focal_image, feature_extractor, transform)

        for thumbnail, (thumb_id, label) in zip(focal_thumbnails, annotations.items()):
            thumbnail_features = extract_features(
                thumbnail, feature_extractor, transform
            )
            similarity = cosine_similarity(
                focal_features.unsqueeze(0), thumbnail_features.unsqueeze(0)
            ).item()
            # print(label)
            data.append(
                {
                    "focal_image_id": focal_image_id,
                    "similarity": similarity,
                    "true_label": label,  # annotation["label"],
                    "thumbnail_id": thumb_id,
                    # "thumbnail_url": annotation["thumbnail_url"]
                }
            )

    return pd.DataFrame(data)


import torch





process_all_images()

# %%
base_dir = Path("/Users/vigji/My Drive/eco-beauty")
feature_extractor, transform, _ = setup_image_processing()

df = create_similarity_dataframe_gpu(base_dir, feature_extractor, transform)
df["true_label_int"] = df["true_label"].apply(lambda x: 1 if x == "same" else 0)
# Save the dataframe to a CSV file
output_dir = base_dir / "google_lens_search/analysis"
output_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(output_dir / "similarity_scores.csv", index=False)
print(f"Saved similarity scores to {output_dir / 'similarity_scores.csv'}")

# %%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

df["true_label_int"] = df["true_label"].apply(lambda x: 1 if x == "same" else 0)


def find_best_threshold(df, min_accuracy=0.9):
    """
    Find the best threshold for separating 'same' and 'different' distributions.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'similarity' and 'true_label' columns.
    min_accuracy : float, optional
        Minimum accuracy to achieve, by default 0.9.

    Returns
    -------
    float
        Best threshold value.
    float
        Accuracy achieved with the best threshold.
    """
    thresholds = np.arange(0, 1.01, 0.01)
    predicted = (df["similarity"].values[:, np.newaxis] >= thresholds).astype(int)
    # predicted = predicted.apply(lambda x: "same" if x == 1 else "different", axis=1)

    accuracies = np.apply_along_axis(
        accuracy_score, 0, df["true_label_int"].values[:, np.newaxis], predicted
    )

    valid_thresholds = accuracies >= min_accuracy
    if np.any(valid_thresholds):
        best_index = np.argmax(accuracies * valid_thresholds)
        best_threshold = thresholds[best_index]
        best_accuracy = accuracies[best_index]
    else:
        best_threshold = 0
        best_accuracy = 0

    return best_threshold, best_accuracy


# Create the plot
plt.figure(figsize=(12, 6))

# Subplot for 'same' distribution
plt.subplot(1, 2, 1)
sns.swarmplot(data=df, x="true_label", y="similarity", color="blue")
plt.title("'Same' Distribution")
plt.xlabel("True Label (1 = Same)")
plt.ylabel("Similarity Score")


plt.tight_layout()

# Find best threshold
# best_threshold, best_accuracy = find_best_threshold(df)

# print(f"Best threshold: {best_threshold:.2f}")
# print(f"Accuracy achieved: {best_accuracy:.2%}")

# Add threshold line to both subplots
for i in range(1, 3):
    plt.subplot(1, 2, i)
    # plt.axhline(y=best_threshold, color='green', linestyle='--', label=f'Threshold ({best_threshold:.2f})')
    plt.legend()

plt.show()

# %%
df
# %%
