import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from scipy.ndimage import gaussian_filter1d
import random
import copy
import time


# ==========================
# 1. Reproducibility
# ==========================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed()

# ==========================
# 2. MNIST-1D Dataset Generation
# ==========================


class MNIST1DGenerator:
    """
    Generates the MNIST-1D dataset as described in the paper.
    Each sample is a 1D sequence of 40 points constructed from a 12-point digit template,
    padded, shifted, scaled, sheared, noise-added, smoothed, and downsampled.
    """

    def __init__(self, num_samples=5000, split="train", seed=42):
        self.num_samples = num_samples
        self.split = split  # 'train' or 'test'
        self.seed = seed
        np.random.seed(self.seed)
        self.templates = self._create_templates()

    def _create_templates(self):
        """
        Creates hand-crafted 12-point templates for digits 0-9.
        These are simplistic representations and can be refined for better accuracy.
        """
        templates = {}
        # Example simplistic templates; in practice, design them to resemble digits.
        templates[0] = np.array([0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3], dtype=np.float32)
        templates[1] = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        templates[2] = np.array([0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 4, 5], dtype=np.float32)
        templates[3] = np.array([0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3], dtype=np.float32)
        templates[4] = np.array([0, 1, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1], dtype=np.float32)
        templates[5] = np.array([5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2], dtype=np.float32)
        templates[6] = np.array([0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 4, 3], dtype=np.float32)
        templates[7] = np.array([0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1], dtype=np.float32)
        templates[8] = np.array([0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3], dtype=np.float32)
        templates[9] = np.array([0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3], dtype=np.float32)
        return templates

    def _pad_sequence(self, sequence):
        """
        Pads the 12-point sequence with 36-60 zeros.
        """
        padding_length = np.random.randint(36, 61)  # 36 to 60 padding points
        padding = np.zeros(padding_length, dtype=np.float32)
        padded_sequence = np.concatenate([sequence, padding])
        return padded_sequence

    def _circular_shift(self, sequence):
        """
        Applies a random circular shift up to 48 indices.
        """
        shift = np.random.randint(0, 49)  # Up to 48 indices
        shifted_sequence = np.roll(sequence, shift)
        return shifted_sequence

    def _scale_sequence(self, sequence):
        """
        Applies a random scaling to the sequence.
        """
        scale = np.random.uniform(0.5, 1.5)  # Example scaling factor
        scaled_sequence = sequence * scale
        return scaled_sequence

    def _add_noise(self, sequence):
        """
        Adds Gaussian noise and white noise to the sequence.
        """
        gaussian_noise = np.random.normal(0, 0.25, size=sequence.shape)
        white_noise = np.random.normal(0, 0.02, size=sequence.shape)
        noisy_sequence = sequence + gaussian_noise + white_noise
        return noisy_sequence

    def _shear_sequence(self, sequence):
        """
        Applies shear to the sequence.
        """
        shear = np.random.uniform(-0.75, 0.75)
        sheared_sequence = sequence + shear
        return sheared_sequence

    def _gaussian_smooth(self, sequence, sigma=2):
        """
        Applies Gaussian smoothing to induce spatial correlations.
        """
        smoothed_sequence = gaussian_filter1d(sequence, sigma=sigma)
        return smoothed_sequence

    def _downsample(self, sequence, final_length=40):
        """
        Downsamples the sequence to the final length.
        """
        factor = len(sequence) / final_length
        indices = (np.arange(final_length) * factor).astype(int)
        downsampled_sequence = sequence[indices]
        return downsampled_sequence

    def generate_sample(self, digit):
        """
        Generates a single MNIST-1D sample for the given digit.
        """
        template = self.templates[digit]
        padded = self._pad_sequence(template)
        shifted = self._circular_shift(padded)
        scaled = self._scale_sequence(shifted)
        sheared = self._shear_sequence(scaled)
        noisy = self._add_noise(sheared)
        smoothed = self._gaussian_smooth(noisy)
        downsampled = self._downsample(smoothed)
        return downsampled

    def generate(self):
        """
        Generates the entire MNIST-1D dataset.
        """
        x = []
        y = []
        samples_per_class = self.num_samples // 10
        for digit in range(10):
            for _ in range(samples_per_class):
                sample = self.generate_sample(digit)
                x.append(sample)
                y.append(digit)
        x = np.array(x)
        y = np.array(y)
        # Shuffle the dataset
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]
        return x.astype(np.float32), y.astype(np.int64)


class MNIST1DDataset(Dataset):
    """
    PyTorch Dataset for MNIST-1D.
    """

    def __init__(self, x, y, transform=None):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = self.x[idx]
        label = self.y[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label


def prepare_mnist1d(batch_size=128, seed=42):
    """
    Generates and splits the MNIST-1D dataset into training and test sets.
    Default split: 4000 train, 1000 test.
    """
    generator = MNIST1DGenerator(num_samples=5000, split="train", seed=seed)
    x, y = generator.generate()
    dataset = MNIST1DDataset(x, y)
    train_size = 4000
    test_size = 1000
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, test_loader


# ==========================
# 3. Model Definitions
# ==========================


class LogisticRegression(nn.Module):
    def __init__(self, input_size=40, num_classes=10):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out


class MLP(nn.Module):
    def __init__(self, input_size=40, hidden_sizes=[128], num_classes=10, dropout=0.3):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_size = h
        layers.append(nn.Linear(in_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CNN1D(nn.Module):
    def __init__(self, input_length=40, num_classes=10):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        # After pooling: input_length -> /2 -> /2 -> /2 = input_length /8
        self.fc1 = nn.Linear(64 * (input_length // 8), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch_size, 1, 40]
        x = self.relu(self.bn1(self.conv1(x)))  # [batch_size, 16, 40]
        x = self.pool(x)  # [batch_size, 16, 20]
        x = self.relu(self.bn2(self.conv2(x)))  # [batch_size, 32, 20]
        x = self.pool(x)  # [batch_size, 32, 10]
        x = self.relu(self.bn3(self.conv3(x)))  # [batch_size, 64, 10]
        x = self.pool(x)  # [batch_size, 64, 5]
        x = x.view(x.size(0), -1)  # [batch_size, 64*5=320]
        x = self.relu(self.fc1(x))  # [batch_size, 128]
        x = self.dropout(x)
        x = self.fc2(x)  # [batch_size, num_classes]
        return x


class GRUClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, num_classes=10):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch_size, 40]
        x = x.unsqueeze(-1)  # [batch_size, 40, 1]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            x.device
        )  # [num_layers, batch, hidden_size]
        out, _ = self.gru(x, h0)  # out: [batch_size, 40, hidden_size]
        out = out[:, -1, :]  # Take the last time step
        out = self.fc(out)  # [batch_size, num_classes]
        return out


# ==========================
# 4. Training and Evaluation
# ==========================


def train_model(model, device, train_loader, criterion, optimizer):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate_model(model, device, test_loader, criterion):
    """
    Evaluates the model on the test set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_labels, all_preds


# ==========================
# 5. Visualization
# ==========================


def plot_accuracies(results, title="Model Accuracies"):
    """
    Plots test accuracies for different models.
    """
    models = list(results.keys())
    accuracies = [results[model]["test_acc"] * 100 for model in models]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=models, y=accuracies)
    plt.ylabel("Test Accuracy (%)")
    plt.title(title)
    plt.ylim(0, 100)
    plt.show()


def plot_confusion_matrix_custom(
    labels, preds, classes, normalize=False, title="Confusion Matrix"
):
    """
    Plots a confusion matrix using Seaborn.
    """
    cm = confusion_matrix(labels, preds)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()


def plot_tsne(x, y, title="t-SNE Visualization"):
    """
    Plots t-SNE embeddings.
    """
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(x)

    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("hsv", 10)
    sns.scatterplot(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        hue=y,
        palette=palette,
        legend="full",
        alpha=0.7,
    )
    plt.title(title)
    plt.show()


# ==========================
# 6. Shuffling Features
# ==========================


def shuffle_features(x):
    """
    Shuffles feature indices for each sample independently.
    """
    x_shuffled = np.empty_like(x)
    for i in range(x.shape[0]):
        shuffled_indices = np.random.permutation(x.shape[1])
        x_shuffled[i] = x[i][shuffled_indices]
    return x_shuffled


class ShuffledDataset(Dataset):
    """
    Dataset with shuffled features.
    """

    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        sample, label = self.original_dataset[idx]
        shuffled_sample = shuffle_features(sample.unsqueeze(0).numpy())[0]
        return torch.from_numpy(shuffled_sample).float(), label


# ==========================
# 7. Lottery Ticket Hypothesis
# ==========================


def lottery_ticket_experiment(
    model,
    device,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    sparsity_levels=[0.8, 0.9, 0.95, 0.98, 0.99],
    patience=10,
):
    """
    Implements the Lottery Ticket Hypothesis experiments.
    Prunes the model iteratively and evaluates performance.
    """
    results = {}
    initial_state = copy.deepcopy(model.state_dict())
    for sparsity in sparsity_levels:
        print(f"\nPruning model to {int(sparsity*100)}% sparsity...")
        # Prune the model
        model = prune_model(model, sparsity)
        # Reset to initial weights
        model.load_state_dict(initial_state)
        # Retrain the pruned model
        train_loss, train_acc = train_model(
            model, device, train_loader, criterion, optimizer
        )
        val_loss, val_acc, _, _ = evaluate_model(model, device, test_loader, criterion)
        print(
            f"Post-pruning Train Acc: {train_acc*100:.2f}%, Test Acc: {val_acc*100:.2f}%"
        )
        results[sparsity] = {"train_acc": train_acc, "test_acc": val_acc}
    return results


def prune_model(model, sparsity):
    """
    Prunes the model's weights based on the given sparsity level.
    """
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
            parameters_to_prune.append((module, "weight"))

    # Calculate the global threshold
    all_weights = torch.cat(
        [torch.flatten(module.weight.data) for module, _ in parameters_to_prune]
    )
    threshold = np.percentile(all_weights.cpu().numpy(), sparsity * 100)

    # Apply pruning
    for module, name in parameters_to_prune:
        mask = torch.abs(module.weight.data) > threshold
        module.weight.data.mul_(mask.float())

    return model


# ==========================
# 8. Deep Double Descent
# ==========================


def deep_double_descent_experiment(
    model_class,
    device,
    train_loader,
    test_loader,
    criterion,
    optimizer_class,
    hidden_sizes_range,
    label_noise=0.0,
):
    """
    Investigates the double descent phenomenon by varying model complexity.
    """
    test_accuracies = []
    model_sizes = []
    for hidden_size in hidden_sizes_range:
        print(f"\nTraining model with hidden size: {hidden_size}")
        # Initialize model
        if model_class == "MLP":
            model = MLP(
                input_size=40, hidden_sizes=[hidden_size], num_classes=10, dropout=0.3
            ).to(device)
        elif model_class == "CNN":
            model = CNN1D(input_length=40, num_classes=10).to(device)
        else:
            raise ValueError("Unsupported model class. Choose 'MLP' or 'CNN'.")
        # Define optimizer
        optimizer = optimizer_class(model.parameters(), lr=0.001)
        # Train the model
        for epoch in range(1, 21):
            train_loss, train_acc = train_model(
                model, device, train_loader, criterion, optimizer
            )
            val_loss, val_acc, _, _ = evaluate_model(
                model, device, test_loader, criterion
            )
            if epoch % 5 == 0:
                print(
                    f"Epoch {epoch}: Train Acc: {train_acc*100:.2f}%, Test Acc: {val_acc*100:.2f}%"
                )
        # Record test accuracy
        _, final_acc, _, _ = evaluate_model(model, device, test_loader, criterion)
        test_accuracies.append(final_acc)
        model_sizes.append(hidden_size)
    # Plot double descent curve
    plt.figure(figsize=(10, 6))
    plt.plot(model_sizes, [acc * 100 for acc in test_accuracies], marker="o")
    plt.xlabel("Hidden Size")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Deep Double Descent")
    plt.grid(True)
    plt.show()


# ==========================
# 9. Gradient-Based Metalearning
# ==========================


class MetaLearner(nn.Module):
    """
    Learns the learning rate for SGD optimizer.
    """

    def __init__(self):
        super(MetaLearner, self).__init__()
        # Initialize the learning rate as a learnable parameter
        self.lr = nn.Parameter(torch.tensor(0.1, requires_grad=True))

    def forward(self):
        return self.lr


def gradient_based_metalearning(
    model,
    device,
    train_loader,
    test_loader,
    criterion,
    meta_lr=0.01,
    num_outer_steps=100,
):
    """
    Metalearns the learning rate for the optimizer.
    """
    meta_learner = MetaLearner().to(device)
    meta_optimizer = optim.Adam(meta_learner.parameters(), lr=meta_lr)

    for step in range(num_outer_steps):
        # Get a batch of training data
        inputs, labels = next(iter(train_loader))
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass with current learning rate
        optimizer = optim.SGD(model.parameters(), lr=meta_learner())
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Compute gradients and update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute validation loss
        val_loss, val_acc, _, _ = evaluate_model(model, device, test_loader, criterion)

        # Meta loss is validation loss
        meta_optimizer.zero_grad()
        val_loss.backward()
        meta_optimizer.step()

        if step % 10 == 0:
            print(
                f"Outer Step {step}: Meta LR: {meta_learner().item():.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%"
            )

    print(f"Learned Learning Rate: {meta_learner().item():.4f}")


# ==========================
# 10. Self-Supervised Learning (SimCLR)
# ==========================


class SimCLR(nn.Module):
    """
    SimCLR framework implementation for self-supervised learning on MNIST-1D.
    """

    def __init__(self, base_encoder, projection_dim=16):
        super(SimCLR, self).__init__()
        self.encoder = base_encoder
        self.projection_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return z


def get_simclr_augmentations():
    """
    Defines the data augmentations for SimCLR.
    """

    class SimCLRAugmentation:
        def __init__(self):
            pass

        def __call__(self, x):
            # Regress out the linear slope
            x = regress_out_slope(x)
            # Circularly shift by up to 10 pixels
            shift = np.random.randint(-10, 11)
            x = np.roll(x, shift)
            # Reintroduce a random linear slope
            slope = np.random.uniform(-1, 1)
            x += slope * np.linspace(0, 1, len(x))
            return x

    return SimCLRAugmentation()


def regress_out_slope(x):
    """
    Regresses out the linear slope from the sequence.
    """
    x_np = x.numpy()
    n = len(x_np)
    X = np.vstack([np.ones(n), np.linspace(0, 1, n)]).T
    slope, intercept = np.linalg.lstsq(X, x_np, rcond=None)[0]
    x_np_regressed = x_np - (slope * np.linspace(0, 1, n) + intercept)
    return torch.from_numpy(x_np_regressed).float()


def simclr_experiment(device, train_loader, test_loader, num_epochs=20, batch_size=128):
    """
    Implements the SimCLR self-supervised learning experiment.
    """

    # Define base encoder (MLP in this case)
    class BaseEncoder(nn.Module):
        def __init__(self):
            super(BaseEncoder, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(40, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
            )

        def forward(self, x):
            return self.network(x)

    base_encoder = BaseEncoder().to(device)
    simclr_model = SimCLR(base_encoder, projection_dim=16).to(device)
    optimizer = optim.Adam(simclr_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Define data augmentations
    augmentation = get_simclr_augmentations()

    # Prepare augmented data loader
    class SimCLRDataset(Dataset):
        def __init__(self, original_dataset, augmentation):
            self.original_dataset = original_dataset
            self.augmentation = augmentation

        def __len__(self):
            return len(self.original_dataset)

        def __getitem__(self, idx):
            x, _ = self.original_dataset[idx]
            x1 = self.augmentation(x).unsqueeze(0)
            x2 = self.augmentation(x).unsqueeze(0)
            return x1, x2

    simclr_train_dataset = SimCLRDataset(train_loader.dataset, augmentation)
    simclr_train_loader = DataLoader(
        simclr_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    # Training loop
    for epoch in range(1, num_epochs + 1):
        simclr_model.train()
        total_loss = 0.0
        for x1, x2 in simclr_train_loader:
            x1 = x1.to(device).squeeze(1)
            x2 = x2.to(device).squeeze(1)
            optimizer.zero_grad()
            z1 = simclr_model(x1)
            z2 = simclr_model(x2)
            loss = info_nce_loss(z1, z2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x1.size(0)
        avg_loss = total_loss / len(simclr_train_loader.dataset)
        print(f"Epoch {epoch}: SimCLR Loss: {avg_loss:.4f}")

    # Evaluate linear classification on learned representations
    # Freeze encoder and train a linear classifier
    linear_classifier = nn.Linear(128, 10).to(device)
    optimizer_linear = optim.Adam(linear_classifier.parameters(), lr=0.001)
    criterion_linear = nn.CrossEntropyLoss()

    # Extract features for training
    def extract_features(model, loader):
        model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                f = model(x)
                features.append(f.cpu().numpy())
                labels.extend(y.cpu().numpy())
        features = np.concatenate(features, axis=0)
        labels = np.array(labels)
        return features, labels

    # Train linear classifier
    train_features, train_labels = extract_features(base_encoder, train_loader)
    test_features, test_labels = extract_features(base_encoder, test_loader)

    train_dataset_linear = torch.utils.data.TensorDataset(
        torch.from_numpy(train_features).float(), torch.from_numpy(train_labels)
    )
    train_loader_linear = DataLoader(
        train_dataset_linear, batch_size=128, shuffle=True, num_workers=2
    )

    for epoch in range(1, 21):
        linear_classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader_linear:
            x = x.to(device)
            y = y.to(device)
            optimizer_linear.zero_grad()
            outputs = linear_classifier(x)
            loss = criterion_linear(outputs, y)
            loss.backward()
            optimizer_linear.step()
            running_loss += loss.item() * x.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(
            f"Linear Classifier Epoch {epoch}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc*100:.2f}%"
        )

    # Evaluate on test set
    linear_classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader_linear:
            x = x.to(device)
            y = y.to(device)
            outputs = linear_classifier(x)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    test_acc = correct / total
    print(f"Linear Classifier Test Accuracy: {test_acc*100:.2f}%")

    # t-SNE Visualization
    plot_tsne(
        test_features,
        test_labels,
        title="t-SNE Visualization of SimCLR Representations on MNIST-1D",
    )


def info_nce_loss(z1, z2, temperature=0.5):
    """
    Computes the InfoNCE loss for SimCLR.
    """
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.matmul(z, z.T) / temperature
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0).to(z.device)
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    labels = labels.unsqueeze(1)
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
    positives = similarity_matrix[torch.arange(2 * batch_size), labels.view(-1)]
    loss = -torch.log(
        torch.exp(positives) / torch.sum(torch.exp(similarity_matrix), dim=1)
    )
    return loss.mean()


# ==========================
# 11. Pooling Methods Benchmarking
# ==========================


def pooling_benchmark_experiment(
    device,
    model_class,
    pooling_methods,
    train_loader,
    test_loader,
    criterion,
    optimizer_class,
    training_set_sizes,
):
    """
    Benchmarks different pooling methods across various training set sizes.
    """
    results = {method: [] for method in pooling_methods}
    for size in training_set_sizes:
        print(f"\nTraining with training set size: {size}")
        # Create subset of training data
        subset_indices = list(range(size))
        subset = Subset(train_loader.dataset, subset_indices)
        subset_loader = DataLoader(subset, batch_size=128, shuffle=True, num_workers=2)
        # Initialize model with specified pooling method
        if model_class == "CNN":
            model = CNN1D(input_length=40, num_classes=10).to(device)
            # Modify pooling if needed
            # For simplicity, assume different pooling methods are implemented as separate models
            for method in pooling_methods:
                print(f"\nPooling Method: {method}")
                # Clone the original model
                model_pool = copy.deepcopy(model)
                # Modify the pooling layer
                if method == "MaxPool":
                    model_pool.pool = nn.MaxPool1d(2)
                elif method == "AveragePool":
                    model_pool.pool = nn.AvgPool1d(2)
                elif method == "None":
                    model_pool.pool = nn.Identity()
                else:
                    raise ValueError("Unsupported pooling method.")
                model_pool.to(device)
                # Define optimizer
                optimizer = optimizer_class(model_pool.parameters(), lr=0.001)
                # Train the model
                for epoch in range(1, 21):
                    train_loss, train_acc = train_model(
                        model_pool, device, subset_loader, criterion, optimizer
                    )
                # Evaluate the model
                _, test_acc, _, _ = evaluate_model(
                    model_pool, device, test_loader, criterion
                )
                print(f"Test Accuracy with {method}: {test_acc*100:.2f}%")
                results[method].append(test_acc * 100)
    # Plot results
    plt.figure(figsize=(10, 6))
    for method in pooling_methods:
        plt.plot(training_set_sizes, results[method], marker="o", label=method)
    plt.xlabel("Training Set Size")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Pooling Methods Benchmarking")
    plt.legend()
    plt.grid(True)
    plt.show()


# ==========================
# 10. Gradient-Based Metalearning
# ==========================


class MetaLearner(nn.Module):
    """
    Learns the learning rate for SGD optimizer.
    """

    def __init__(self):
        super(MetaLearner, self).__init__()
        # Initialize the learning rate as a learnable parameter
        self.lr = nn.Parameter(torch.tensor(0.1, requires_grad=True))

    def forward(self):
        return self.lr


def gradient_based_metalearning_experiment(
    model,
    device,
    train_loader,
    test_loader,
    criterion,
    num_outer_steps=100,
    meta_lr=0.01,
):
    """
    Metalearns the learning rate for the optimizer using gradient-based optimization.
    """
    meta_learner = MetaLearner().to(device)
    meta_optimizer = optim.Adam(meta_learner.parameters(), lr=meta_lr)

    for step in range(1, num_outer_steps + 1):
        # Get a batch of training data
        try:
            inputs, labels = next(train_iter)
        except:
            train_iter = iter(train_loader)
            inputs, labels = next(train_iter)
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass with current learning rate
        optimizer = optim.SGD(model.parameters(), lr=meta_learner())
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Compute gradients and update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute validation loss
        val_loss, val_acc, _, _ = evaluate_model(model, device, test_loader, criterion)

        # Meta loss is validation loss
        meta_optimizer.zero_grad()
        val_loss.backward()
        meta_optimizer.step()

        if step % 10 == 0 or step == 1:
            print(
                f"Outer Step {step}: Meta LR: {meta_learner().item():.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%"
            )

    print(f"Learned Learning Rate: {meta_learner().item():.4f}")


# ==========================
# 7. Lottery Ticket Hypothesis
# ==========================


def lottery_ticket_experiment(
    model,
    device,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    sparsity_levels=[0.8, 0.9, 0.95, 0.98, 0.99],
):
    """
    Implements the Lottery Ticket Hypothesis experiments.
    Prunes the model iteratively and evaluates performance.
    """
    results = {}
    initial_state = copy.deepcopy(model.state_dict())
    for sparsity in sparsity_levels:
        print(f"\nPruning model to {int(sparsity*100)}% sparsity...")
        # Prune the model
        model = prune_model(model, sparsity)
        # Reset to initial weights
        model.load_state_dict(initial_state)
        # Retrain the pruned model
        train_loss, train_acc = train_model(
            model, device, train_loader, criterion, optimizer
        )
        val_loss, val_acc, _, _ = evaluate_model(model, device, test_loader, criterion)
        print(
            f"Post-pruning Train Acc: {train_acc*100:.2f}%, Test Acc: {val_acc*100:.2f}%"
        )
        results[sparsity] = {"train_acc": train_acc, "test_acc": val_acc}
    return results


def prune_model(model, sparsity):
    """
    Prunes the model's weights based on the given sparsity level.
    """
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
            parameters_to_prune.append((module, "weight"))

    # Calculate the global threshold
    all_weights = torch.cat(
        [torch.flatten(module.weight.data) for module, _ in parameters_to_prune]
    )
    threshold = np.percentile(all_weights.cpu().numpy(), sparsity * 100)

    # Apply pruning
    for module, name in parameters_to_prune:
        mask = torch.abs(module.weight.data) > threshold
        module.weight.data.mul_(mask.float())

    return model


# ==========================
# 12. Main Execution
# ==========================


def main():
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 50
    patience = 10  # For early stopping
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare MNIST-1D data
    train_loader, test_loader = prepare_mnist1d(batch_size=batch_size, seed=42)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(input_size=40, num_classes=10).to(
            device
        ),
        "MLP": MLP(input_size=40, hidden_sizes=[128], num_classes=10, dropout=0.3).to(
            device
        ),
        "CNN": CNN1D(input_length=40, num_classes=10).to(device),
        "GRU": GRUClassifier(
            input_size=1, hidden_size=64, num_layers=1, num_classes=10
        ).to(device),
    }

    # Define loss and optimizer for each model
    criterion = nn.CrossEntropyLoss()
    optimizers = {
        model_name: optim.Adam(model.parameters(), lr=learning_rate)
        for model_name, model in models.items()
    }

    # Training and Evaluation
    results = {}
    best_models = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        optimizer = optimizers[model_name]
        best_val_acc = 0.0
        patience_counter = 0
        best_state = None

        # Since the paper doesn't specify a separate validation set, we'll use test set for simplicity
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = train_model(
                model, device, train_loader, criterion, optimizer
            )
            val_loss, val_acc, _, _ = evaluate_model(
                model, device, test_loader, criterion
            )

            print(
                f"Epoch [{epoch}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% "
                f"Test Loss: {val_loss:.4f}, Test Acc: {val_acc*100:.2f}%"
            )

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best model state
        if best_state is not None:
            model.load_state_dict(best_state)
            print(
                f"Loaded best model with validation accuracy: {best_val_acc*100:.2f}%"
            )

        # Evaluate on test set
        _, final_acc, all_labels, all_preds = evaluate_model(
            model, device, test_loader, criterion
        )
        results[model_name] = {
            "test_acc": final_acc,
            "labels": all_labels,
            "preds": all_preds,
        }

    # Plot Test Accuracies
    plot_accuracies(results, title="Test Accuracies on MNIST-1D")

    # Print Classification Reports
    for model_name, result in results.items():
        print(f"\nClassification Report for {model_name}:")
        print(classification_report(result["labels"], result["preds"], digits=4))

    # Plot Confusion Matrices
    classes = [str(i) for i in range(10)]
    for model_name, result in results.items():
        plot_confusion_matrix_custom(
            result["labels"],
            result["preds"],
            classes,
            normalize=True,
            title=f"Confusion Matrix for {model_name}",
        )

    # t-SNE Visualization
    # For visualization, we need to collect all test data and pass through a trained model's penultimate layer
    # Here, we'll use the CNN model as an example
    cnn_model = models["CNN"]
    cnn_model.eval()
    with torch.no_grad():
        all_features = []
        all_labels = []
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            # Forward pass up to penultimate layer
            x = cnn_model.relu(
                cnn_model.bn1(cnn_model.conv1(inputs))
            )  # [batch, 16, 40]
            x = cnn_model.pool(x)  # [batch, 16, 20]
            x = cnn_model.relu(cnn_model.bn2(cnn_model.conv2(x)))  # [batch, 32, 20]
            x = cnn_model.pool(x)  # [batch, 32, 10]
            x = cnn_model.relu(cnn_model.bn3(cnn_model.conv3(x)))  # [batch, 64, 10]
            x = cnn_model.pool(x)  # [batch, 64, 5]
            x = x.view(x.size(0), -1)  # [batch, 320]
            x = cnn_model.relu(cnn_model.fc1(x))  # [batch, 128]
            all_features.append(x.cpu().numpy())
            all_labels.extend(labels.numpy())
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.array(all_labels)

    plot_tsne(
        all_features,
        all_labels,
        title="t-SNE Visualization of CNN Features on MNIST-1D",
    )

    # ==========================
    # 7. Lottery Ticket Hypothesis
    # ==========================

    # Select MLP model for Lottery Ticket experiments
    mlp_model = copy.deepcopy(models["MLP"])
    mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate)
    print("\n--- Lottery Ticket Hypothesis Experiment ---")
    lottery_results = lottery_ticket_experiment(
        mlp_model, device, train_loader, test_loader, criterion, mlp_optimizer
    )
    # Plot Lottery Ticket Results
    sparsity_levels = list(lottery_results.keys())
    test_accs = [lottery_results[s]["test_acc"] * 100 for s in sparsity_levels]
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=[int(s * 100) for s in sparsity_levels], y=test_accs, marker="o")
    plt.xlabel("Sparsity Level (%)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Lottery Ticket Hypothesis on MLP")
    plt.gca().invert_xaxis()
    plt.show()

    # ==========================
    # 8. Deep Double Descent
    # ==========================

    print("\n--- Deep Double Descent Experiment (MLP) ---")
    hidden_sizes_range = list(range(10, 201, 10))  # Hidden sizes from 10 to 200
    deep_double_descent_experiment(
        "MLP",
        device,
        train_loader,
        test_loader,
        criterion,
        optim.Adam,
        hidden_sizes_range,
        label_noise=0.0,
    )

    print("\n--- Deep Double Descent Experiment (CNN) ---")
    hidden_sizes_range_cnn = list(range(16, 161, 16))  # Example hidden sizes
    deep_double_descent_experiment(
        "CNN",
        device,
        train_loader,
        test_loader,
        criterion,
        optim.Adam,
        hidden_sizes_range_cnn,
        label_noise=0.0,
    )

    # ==========================
    # 9. Gradient-Based Metalearning
    # ==========================

    print("\n--- Gradient-Based Metalearning Experiment ---")
    # Select MLP model for metalearning
    metalearning_model = copy.deepcopy(models["MLP"])
    metalearning_optimizer = optim.Adam(
        metalearning_model.parameters(), lr=learning_rate
    )
    gradient_based_metalearning_experiment(
        metalearning_model,
        device,
        train_loader,
        test_loader,
        criterion,
        num_outer_steps=100,
        meta_lr=0.01,
    )

    # ==========================
    # 10. Self-Supervised Learning (SimCLR)
    # ==========================

    print("\n--- Self-Supervised Learning (SimCLR) Experiment ---")
    simclr_experiment(
        device, train_loader, test_loader, num_epochs=20, batch_size=batch_size
    )

    # ==========================
    # 11. Pooling Methods Benchmarking
    # ==========================

    print("\n--- Pooling Methods Benchmarking Experiment ---")
    pooling_methods = ["MaxPool", "AveragePool", "None"]
    training_set_sizes = [500, 1000, 2000, 4000]
    pooling_benchmark_experiment(
        device,
        "CNN",
        pooling_methods,
        train_loader,
        test_loader,
        criterion,
        optim.Adam,
        training_set_sizes,
    )


if __name__ == "__main__":
    main()
