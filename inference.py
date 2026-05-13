import torch

from torchvision import transforms, models
from PIL import Image

# -----------------------------------
# Device
# -----------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

# -----------------------------------
# Class Names
# -----------------------------------

class_names = [
    'buildings',
    'forest',
    'glacier',
    'mountain',
    'sea',
    'street'
]

# -----------------------------------
# Load Model
# -----------------------------------

model = models.resnet18(weights=None)

num_features = model.fc.in_features

model.fc = torch.nn.Linear(
    num_features,
    len(class_names)
)

model.load_state_dict(
    torch.load(
        "models/resnet18_best.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model Loaded Successfully")

# -----------------------------------
# Transform
# -----------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -----------------------------------
# Load Image
# -----------------------------------

image = Image.open("sample.jpg").convert("RGB")

image = transform(image)

image = image.unsqueeze(0)

image = image.to(device)

# -----------------------------------
# Prediction
# -----------------------------------

with torch.no_grad():

    outputs = model(image)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted = torch.max(
        probabilities,
        1
    )

predicted_class = class_names[predicted.item()]

confidence_score = confidence.item() * 100

print("\nPrediction:", predicted_class)

print(f"Confidence: {confidence_score:.2f}%")