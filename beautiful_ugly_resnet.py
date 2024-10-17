# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import shutil
from random import sample
from tqdm import tqdm

prepare_data = False
split_ratio = 0.2
data_dir = Path("/Users/vigji/My Drive/eco-beauty/dataset/resnet_balanced")
device = "mps"
categories = ["beautiful", "ugly"]

# split the data into training and validation sets folders:

# %%
source_data_dir = Path("/Users/vigji/Downloads/train")
cat_files = list(data_dir.glob(f"*{categories[0]}*"))
dog_files = list(data_dir.glob(f"*{categories[1]}*"))
if prepare_data:
    # Split the data into training and validation sets
    for category in categories:
        cat_files = list(source_data_dir.glob(f"*{category}*"))
        # random draw of validation images
        val_files = sample(cat_files, int(len(cat_files) * split_ratio))
        train_files = set(cat_files) - set(val_files)

        for group, files in [("train", train_files), ("val", val_files)]:
            current_data_dir = data_dir / group / category
            current_data_dir.mkdir(exist_ok=True, parents=True)
            for file in tqdm(files):
                shutil.copy(file, current_data_dir / file.name)

# %%
# Data augmentation and normalization for training, normalization for validation
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

image_datasets = {
    x: datasets.ImageFolder(f"{data_dir}/{x}", data_transforms[x])
    for x in ["train", "val"]
}
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=32, shuffle=True)
    for x in ["train", "val"]
}

dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
class_names = image_datasets["train"].classes
device = torch.device(device)

# %%
# Load pre-trained ResNet model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Modify the last layer to classify into 2 classes (cats and dogs)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Move the model to the appropriate device
model = model.to(torch.float32)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# %%
def train_model(model, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(torch.float32).to(device)
                labels = labels.to(torch.float32).to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    return model


# Train the model
if __name__ == "__main__":
    model = train_model(model, criterion, optimizer, num_epochs=100)
    torch.save(model.state_dict(), "beautiful_vs_ugly_resnet18.pth")

# %%
