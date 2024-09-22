# %%
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


from pathlib import Path
import shutil
from random import sample

print("Splitting the dataset keeping out validation images...")
# Set the paths
base_dir = Path('/Users/vigji/My Drive/eco-beauty/dataset')
categories = ['beautiful', 'ugly']  # Subdirectories in the dataset
raw_dir = base_dir / 'raw'
train_dir = base_dir / 'model_fit'
val_dir = base_dir / 'validation_leftout'

# Create training and validation directories if they don't exist
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

# Define the splitting ratio
validation_ratio = 0.1

# Process each category
for category in categories:
    source_dir = raw_dir / category
    images = list(source_dir.glob('*.png'))  # Get all files in the directory
    print(base_dir / category, source_dir.exists())
    num_validation = int(len(images) * validation_ratio)
    
    # Randomly select images for validation
    validation_images = sample(images, num_validation)
    # exclude validation images from the training set:
    fit_images = set(images) - set(validation_images)
    
    for image_paths, directory in [(fit_images, train_dir), (validation_images, val_dir)]:  # Loop over the images
        destination_dir = directory / category
        destination_dir.mkdir(parents=True, exist_ok=True)
        for image_path in image_paths:
            shutil.copy(image_path, destination_dir / image_path.name)

base_dir

# %%
# Load an image using PIL
img = Image.open('/Users/vigji/My Drive/eco-beauty/dataset/raw/beautiful/img011.png')

# Define the transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),  # Color jitter
    transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),  # Random crop and resize (zoom effect)
    transforms.ToTensor()
])

# Apply the transformations
augmented_img = transform(img)

# Convert tensor back to PIL image for visualization
augmented_img_pil = transforms.ToPILImage()(augmented_img)


# Plot the original and augmented images
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Original image
axs[0].imshow(img)
axs[0].set_title('Original Image')
axs[0].axis('off')

# Augmented image
axs[1].imshow(augmented_img_pil)
axs[1].set_title('Augmented Image')
axs[1].axis('off')

plt.show()

# %%

print("Augmenting the dataset...")
# Number of augmentations per image
augmentation_factor = 10

augmented_dataset_dir = train_dir / 'augmented'

for category in categories:
    source_dir = train_dir / category
    destination_dir = augmented_dataset_dir / category
    destination_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in tqdm(list(source_dir.glob('*.png'))):
        img = Image.open(img_path)
        
        for i in range(augmentation_factor):
            augmented_img = transform(img)
            augmented_img_pil = transforms.ToPILImage()(augmented_img)
            
            output_file_path = destination_dir / f"{img_path.stem}_aug_{i}{img_path.suffix}"
            augmented_img_pil.save(output_file_path)

# %%
