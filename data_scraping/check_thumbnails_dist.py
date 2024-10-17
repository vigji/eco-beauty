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


focal_image_id = "beautiful_000"
category, img_number = focal_image_id.split("_")[0], int(focal_image_id.split("_")[1])
focal_image = load_image(category, img_number)
focal_thumbnails = load_thumbnails(category, img_number)

annotations = json.load(open(annotations_dir / f"{focal_image_id}_annotations.json"))
annotations
# %%
# Implement image similarity computation
# Load pre-trained ResNet model
resnet = models.resnet50(pretrained=True)
resnet.eval()

# Remove the last fully connected layer
layers_to_keep = list(resnet.children())[:-1]
feature_extractor = torch.nn.Sequential(*layers_to_keep)

# To compare properly, we first grayscale and normalize:
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
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        #  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def extract_features(image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(img_tensor)
    return features.squeeze()


# %%
# Extract features from focal image
focal_features = extract_features(focal_image)

# %%
# Extract features from thumbnails and compute similarities
similarities = []
for thumbnail in focal_thumbnails:
    thumbnail_features = extract_features(thumbnail)
    similarity = cosine_similarity(
        focal_features.unsqueeze(0), thumbnail_features.unsqueeze(0)
    )
    similarities.append(similarity.item())


# Make a composite image with the thumbnails, sorted by similarity to the focal image
# in a multiple columns format, so that the output is almost square.
# For each, overlay the similarity score as text on top of the thumbnail.
# Make sure thumbnails are sorted by similarity to the focal image.
# %%

# Create a composite image, with 10 columns and as many rows as needed
num_columns = 10
num_rows = (len(focal_thumbnails) + num_columns - 1) // num_columns
composite_image = Image.new("RGB", (256 * num_columns, 256 * num_rows))

# %%
# Create copies of thumbnails with similarity scores overlaid
thumbnails_with_scores = []
for i, (thumbnail, label) in enumerate(zip(focal_thumbnails, annotations.values())):
    thumbnail_copy = thumbnail.copy()
    # apply the same transformation as the one used for feature extraction
    thumbnail_copy = transform_visualization(thumbnail_copy)
    # convert to PIL image, making sure that image is mapped to 0-255 values:
    thumbnail_copy = transforms.ToPILImage()(thumbnail_copy.squeeze(0).clamp(0, 1))

    # Add green frame
    color = "green" if label == "same" else "red"
    framed_thumbnail = Image.new("RGB", thumbnail_copy.size, color=color)
    width_pxs = 4
    # Crop thumbnail_copy to fit within the frame
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

# Sort thumbnails by similarity
sorted_thumbnails_with_scores = [
    x
    for _, x in sorted(
        zip(similarities, thumbnails_with_scores),
        key=lambda pair: pair[0],
        reverse=True,
    )
]

# Paste thumbnails with scores into the composite image
for i, thumbnail in enumerate(sorted_thumbnails_with_scores):
    row = i // num_columns
    col = i % num_columns
    composite_image.paste(thumbnail, (col * 256, row * 256))

# Save the composite image
composite_dir = base_dir / "google_lens_search/composite_images"
composite_dir.mkdir(parents=True, exist_ok=True)
composite_image.save(composite_dir / f"{focal_image_id}_composite.png")


# %%
