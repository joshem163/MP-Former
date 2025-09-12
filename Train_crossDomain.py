import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import warnings

from models import CNNTransformer
from modules import *
from data_loader import *

warnings.filterwarnings("ignore", category=FutureWarning, module="torch_geometric")

# ------------------ Argument parsing ------------------
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--c_channels', type=int, default=64)
parser.add_argument('--d_model', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

# ------------------ Feature Extraction ------------------
def extract_features(dataset):
    list_hks, thres_hks, label = get_thresh_hks(dataset, 10, 0.1)
    list_deg, thres_deg = get_thresh(dataset, 10)

    graph_features = []
    for graph_id in tqdm(range(len(dataset)), desc="Extracting Topological (MP) Features"):
        b0, b1, node, edge = Topo_Fe_TimeSeries_MP(dataset[graph_id],
                                                   list_deg[graph_id], list_hks[graph_id],
                                                   thres_deg, thres_hks)
        graph_features.append(torch.stack([b0, b1, node, edge], dim=0))

    MP_tensor = torch.stack(graph_features)
    labels = torch.tensor(label, dtype=torch.long)
    return MP_tensor, labels


# ------------------ Training Function ------------------
def train_model(model, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in tqdm(range(1, epochs + 1), desc="Training"):
        total_loss, correct, total = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

        acc = correct / total
        tqdm.write(f"Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}, Acc = {acc:.4f}")
    return model


# ------------------ Evaluation Function ------------------
def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            output = model(xb)
            pred = output.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / total


# ------------------ Main ------------------
def main():
    # Pick training and testing datasets manually
    # train_datasets = ["cox2", "bzr","proteins","imdb-binary","ptc"]
    train_datasets = ["ptc", "cox2", "bzr"]
    test_datasets  = ["mutag"]

    # === TRAINING ===
    print("\n=== Extracting Training Data ===")
    X_train_all, y_train_all = [], []
    for dataset_name in train_datasets:
        dataset = load_data(dataset_name)
        X, y = extract_features(dataset)
        X_train_all.append(X)
        y_train_all.append(y)

    X_train = torch.cat(X_train_all, dim=0)
    y_train = torch.cat(y_train_all, dim=0)
    print(f"Train graphs: {len(X_train)}")


    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    num_classes = len(torch.unique(y_train))
    model = CNNTransformer(num_classes=num_classes,
                           cnn_channels=args.c_channels,
                           d_model=args.d_model,
                           drop_out=0.0,
                           nhead=4, num_layers=2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print("\n=== Training Model ===")
    model = train_model(model, train_loader, optimizer, criterion, args.epochs)

    # Save trained model
    model_path = "cnn_transformer.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")

    # === TESTING ===
    print("\n=== Evaluating on Test Data ===")
    X_test_all, y_test_all = [], []
    for dataset_name in test_datasets:
        dataset = load_data(dataset_name)
        X, y = extract_features(dataset)  # Apply Topo_Fe_TimeSeries_MP before testing
        X_test_all.append(X)
        y_test_all.append(y)

    X_test = torch.cat(X_test_all, dim=0)
    y_test = torch.cat(y_test_all, dim=0)
    print(f"Test graphs: {len(X_test)}")
    test_ds = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Load model
    loaded_model = CNNTransformer(num_classes=num_classes,
                                  cnn_channels=args.c_channels,
                                  d_model=args.d_model,
                                  drop_out=0.0,
                                  nhead=4, num_layers=2).to(device)
    #loaded_model.load_state_dict(torch.load(model_path))
    state_dict = torch.load(model_path, weights_only=True)
    loaded_model.load_state_dict(state_dict)

    acc = evaluate_model(loaded_model, test_loader)
    print(f"\n📊 Final Test Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
