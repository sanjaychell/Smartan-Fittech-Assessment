import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

import config

# -----------------------------------
# Device
# -----------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

# -----------------------------------
# Transformations
# -----------------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

# -----------------------------------
# Load Dataset
# -----------------------------------

train_dataset = datasets.ImageFolder(
    config.TRAIN_DIR,
    transform=train_transform
)

class_names = train_dataset.classes

print("Classes:", class_names)

# -----------------------------------
# DataLoader
# -----------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True
)

# -----------------------------------
# MODEL SELECTION
# -----------------------------------

MODEL_NAME = "efficientnet"

# Lightweight Baseline
if MODEL_NAME == "resnet18":

    model = models.resnet18(weights='DEFAULT')

    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        len(class_names)
    )

# Strong Backbone
elif MODEL_NAME == "resnet50":

    model = models.resnet50(weights='DEFAULT')

    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        len(class_names)
    )

# Modern Architecture
elif MODEL_NAME == "efficientnet":

    model = models.efficientnet_b0(weights='DEFAULT')

    num_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        num_features,
        len(class_names)
    )

model = model.to(device)

# -----------------------------------
# Loss Function
# -----------------------------------

criterion = nn.CrossEntropyLoss()

# -----------------------------------
# Optimizer
# -----------------------------------

optimizer = optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE
)

# -----------------------------------
# Training Loop
# -----------------------------------

train_losses = []

for epoch in range(config.EPOCHS):

    print(f"\nEpoch {epoch+1}/{config.EPOCHS}")

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

    train_acc = 100 * correct / total

    train_loss = running_loss / len(train_loader)

    train_losses.append(train_loss)

    print(f"Train Loss: {train_loss:.4f}")

    print(f"Train Accuracy: {train_acc:.2f}%")

# -----------------------------------
# Save Model
# -----------------------------------

torch.save(
    model.state_dict(),
    f"models/{MODEL_NAME}_best.pth"
)

# -----------------------------------
# Plot Loss Curve
# -----------------------------------

plt.plot(train_losses)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training Loss Curve")

plt.savefig(
    "outputs/plots/loss_curve.png"
)

print("\nTraining Completed")