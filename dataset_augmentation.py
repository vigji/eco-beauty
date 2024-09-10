# %%
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# %%
# Load an image using PIL
img = Image.open('/Users/vigji/My Drive/eco-beauty/belli/img011.png')

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