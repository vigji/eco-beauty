# %%
import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the data transforms. Include dropping the alpha channel:
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.Lambda(lambda x: x.convert('RGB') if x.mode == 'RGBA' else x),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the model
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # 2 classes: beautiful and ugly
model.load_state_dict(torch.load('beautiful_vs_ugly_resnet18.pth', map_location=device))
model.to(device)
model.eval()

# Define class names
class_names = ['beautiful', 'ugly']

def predict_image(image_path):
    image = Image.open(image_path)
    image_tensor = data_transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
        
    return class_names[preds[0]], outputs[0]

def plot_prediction(image_path, pred_class, pred_probs):
    image = Image.open(image_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot image
    ax1.imshow(image)
    ax1.set_title(image_path)
    ax1.axis('off')
    
    # Plot probabilities
    probs = torch.nn.functional.softmax(pred_probs, dim=0)
    ax2.bar(class_names, probs.cpu().numpy())
    ax2.set_ylabel('Probability')
    ax2.set_title(f'Prediction: {pred_class}')
    
    plt.show()


# %%
# Example usage
# data_folder = Path("/Users/vigji/My Drive/eco-beauty/dataset/resnet_balanced/val")
data_folder = Path("/Users/vigji/Desktop/google")
beautiful_folder = data_folder / "beautiful"
ugly_folder = data_folder / "ugly"

beautiful_images = sorted(list(beautiful_folder.glob("*.png")))
ugly_images = sorted(list(ugly_folder.glob("*.png")))

# %%
# image_path = 'path/to/your/image.jpg'  # Replace with the path to your image

for image_path in beautiful_images[:10] + ugly_images[:10]:
    pred_class, pred_probs = predict_image(image_path)
    print(f'Predicted class: {pred_class}')
    plot_prediction(image_path, pred_class, pred_probs)
# %%
image_path

# %%
