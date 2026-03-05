import torch
from torch.utils.data import DataLoader, TensorDataset
from custom_model import HandwritingCNN, train_model
import numpy as np
import os

def run_demo_training():
    print("Initializing Custom Handwriting CNN...")
    model = HandwritingCNN(num_classes=10)
    X = torch.randn(100, 1, 28, 28)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10)
    print("Starting Training Loop (From Scratch Implementation)...")
    trained_model = train_model(model, loader, epochs=3)
    torch.save(trained_model.state_dict(), "data/custom_ocr_model.pth")
    print("Model trained and saved to data/custom_ocr_model.pth")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    run_demo_training()
