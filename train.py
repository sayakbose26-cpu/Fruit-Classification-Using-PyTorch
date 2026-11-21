import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


train_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


train_data = datasets.ImageFolder(
    "dataset",
    transform=train_transform
)


train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)


num_classes = len(train_data.classes)

print("Classes:", train_data.classes)
print("Number of classes:", num_classes)

model = models.MobileNetV2(num_classes=num_classes)
model = model.to(device)


loss_func = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)
epochs = 7 


for epoch in range(epochs):

    model.train()

    total_loss = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{epochs}"
    )

    for images, labels in progress_bar:

        images = images.to(device)
        labels = labels.to(device)

        prediction = model(images)

        loss = loss_func(prediction, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


    average_loss = total_loss / len(train_loader)

    print(
        "Epoch:",
        epoch + 1,
        "Average Loss:",
        average_loss
    )


torch.save(
    {
        "classes": train_data.classes,
        "model_state_dict": model.state_dict()
    },
    "fruit_model.pth"
)


print("\nModel successfully saved!")