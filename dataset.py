import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

data_dir = "data/plant_diseases"  # Dataset directory
batch_size = 32
num_classes = 38  # Number of classes
model_save_path = "plant_disease_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transforms for input preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load datasets
data_train = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
data_val = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=transform)

train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False)

# Define the model class
class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def load_model(model_path, num_classes, device):
    model = PlantDiseaseModel(num_classes).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully!")
    else:
        print("No pretrained model found. Using a new model.")
    return model


def main():
    print("Initializing Plant Disease Model")
    model = load_model(model_save_path, num_classes, device)
    print("Model ready for training or evaluation.")

    # Example data check
    sample_data, _ = next(iter(train_loader))
    print(f"Sample input shape: {sample_data.shape}")


if __name__ == "__main__":
    main()
