import torch
import seaborn as sns
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

import config

# -----------------------------------
# Device
# -----------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

# -----------------------------------
# Transform
# -----------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------------
# Load Test Dataset
# -----------------------------------

test_dataset = datasets.ImageFolder(
    config.TEST_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

class_names = test_dataset.classes

print("Classes:", class_names)

# -----------------------------------
# Load Model
# -----------------------------------

model = models.efficientnet_b0(weights=None)

num_features = model.classifier[1].in_features

model.classifier[1] = torch.nn.Linear(
    num_features,
    len(class_names)
)

model.load_state_dict(
    torch.load(
        "models/efficientnet_best.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

# -----------------------------------
# Prediction
# -----------------------------------

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())

        all_labels.extend(labels.numpy())

# -----------------------------------
# Classification Report
# -----------------------------------

print("\nClassification Report:\n")

print(classification_report(
    all_labels,
    all_preds,
    target_names=class_names
))

# -----------------------------------
# Confusion Matrix
# -----------------------------------

cm = confusion_matrix(
    all_labels,
    all_preds
)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.savefig(
    "outputs/confusion_matrix/confusion_matrix.png"
)

print("\nConfusion Matrix Saved")