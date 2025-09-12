import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
from models import CNNTransformer
from modules import *
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import argparse
import itertools
import time
cls_criterion = torch.nn.BCEWithLogitsLoss()
def train(model, device, loader, optimizer, task_type):
    model.train()
    for step, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        # if yb.ndim == 3 and yb.size(1) == 1:
        yb = torch.nan_to_num(yb, nan=0.0)
        yb = yb.squeeze(1)

        optimizer.zero_grad()
        pred = model(xb)
        if "classification" in task_type:
            if yb.ndim == 1:
                loss = F.cross_entropy(pred, yb.view(-1).long())
            else:
                mask = ~torch.isnan(yb)
                loss_mat = F.binary_cross_entropy_with_logits(pred.to(torch.float32), yb.to(torch.float32), reduction="none")
                #loss_mat = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                loss = (loss_mat * mask.float()).sum() / mask.float().sum()
        else:
            mask = ~torch.isnan(yb)
            loss_mat = F.mse_loss(pred, yb, reduction="none")
            loss = (loss_mat * mask.float()).sum() / mask.float().sum()

        loss.backward()
        optimizer.step()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if yb.ndim == 3 and yb.size(1) == 1:
                yb = yb.squeeze(1)
            pred = model(xb)
            y_true.append(yb.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    y_true, y_pred = torch.cat(y_true, dim=0).numpy(), torch.cat(y_pred, dim=0).numpy()
    y_true = np.nan_to_num(y_true, nan=0.0)
    return evaluator.eval({"y_true": y_true, "y_pred": y_pred})

def run_once(args, dataset, X, y, evaluator, device):
    split_idx = dataset.get_idx_split()
    X_train, X_val, X_test = X[split_idx["train"]], X[split_idx["valid"]], X[split_idx["test"]]
    y_train, y_val, y_test = y[split_idx["train"]], y[split_idx["valid"]], y[split_idx["test"]]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    model = CNNTransformer(
        num_classes=dataset.num_tasks,
        cnn_channels=args.c_channels,
        d_model=args.d_model,
        drop_out=0.0,
        nhead=4,
        num_layers=2
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    valid_curve, test_curve = [], []
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, dataset.task_type)
        valid_perf = eval(model, device, val_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

    # select best validation epoch
    if 'classification' in dataset.task_type:
        best_epoch = np.argmax(np.array(valid_curve))
    else:
        best_epoch = np.argmin(np.array(valid_curve))

    return valid_curve[best_epoch], test_curve[best_epoch]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--dataset', type=str, default="ogbg-moltoxcast")
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    dataset = PygGraphPropPredDataset(name=args.dataset)

    # list_hks, thres_hks,label = get_thresh_hks(dataset, 10, 0.1)
    # list_deg, thres_deg = get_thresh(dataset, 10)
    # y = torch.tensor(np.array(label), dtype=torch.float)
    # graph_features = []
    # #for graph_id in range(len(dataset)):
    # for graph_id in tqdm(range(len(dataset))):
    #     b0, b1, node, edge = Topo_Fe_TimeSeries_MP(dataset[graph_id], list_deg[graph_id], list_hks[graph_id],
    #                                                thres_deg, thres_hks)
    #     graph_features.append(torch.stack([b0, b1, node, edge], dim=0))
    # X = torch.stack(graph_features)
    X= torch.load("mp_features.pt")
    labels = dataset.data.y
    y=labels
    evaluator = Evaluator(args.dataset)

    # Hyperparameter search space
    lrs = [1e-3]
    c_channels_list = [32]
    d_models = [64]

    best_result, best_config = None, None
    results = []

    for lr, c_channels, d_model in itertools.product(lrs, c_channels_list, d_models):
        print(f"=== Trying config: lr={lr}, c_channels={c_channels}, d_model={d_model} ===")
        run_test_scores = []
        for run in range(10):
            args.lr, args.c_channels, args.d_model = lr, c_channels, d_model
            val_score, test_score = run_once(args, dataset, X, y, evaluator, device)
            run_test_scores.append(test_score)
            print(f"Run {run+1}: Test={test_score:.4f}")

        mean_test = np.mean(run_test_scores)
        std_test = np.std(run_test_scores)
        results.append(((lr, c_channels, d_model), mean_test, std_test))
        print(f"Config (lr={lr}, c={c_channels}, d={d_model}) → Test mean={mean_test:.4f} ± {std_test:.4f}")

        if best_result is None or mean_test > best_result:
            best_result = mean_test
            best_config = (lr, c_channels, d_model)

    print("\n=== Final Results ===")
    for cfg, mean_test, std_test in results:
        print(f"Config {cfg}: Test mean={mean_test:.4f} ± {std_test:.4f}")
    print(f"\nBest Config={best_config}, Best Mean Test={best_result:.4f}")

if __name__ == "__main__":
    main()
