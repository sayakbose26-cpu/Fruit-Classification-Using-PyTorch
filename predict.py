
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


checkpoints = torch.load(
    "fruit_model.pth",
    map_location=device,
    weights_only=True
)


classes = checkpoints["classes"]

num_classes = len(classes)


model = models.MobileNetV2(
    num_classes=num_classes
)

model.load_state_dict(checkpoints["model_state_dict"])

model = model.to(device)

model.eval()


transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


image = Image.open("images (2).jpg").convert("RGB")

image = transform(image)

image = image.unsqueeze(0).to(device)


with torch.no_grad():

    output = model(image)

    probability = torch.softmax(output,dim=1)

    predict_index = torch.argmax(probability, dim=1).item()

    confidence = probability[0, predict_index].item() * 100


predicted_class = classes[predict_index]


print("\n================================")
print("       FRUIT CLASSIFICATION")
print("================================")

print("Predicted Fruit:", predicted_class)

print("Confidence:", round(confidence, 2), "%")

print("================================")
