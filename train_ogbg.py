import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from models import CNNTransformer
from modules import *
from tqdm import tqdm
import argparse
import time
import numpy as np

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()
def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, (xb, yb) in enumerate(tqdm(loader, desc="Iteration")):
        xb, yb = xb.to(device), yb.to(device)

        # squeeze if extra dimension
        if yb.ndim == 3 and yb.size(1) == 1:
            yb = yb.squeeze(1)

        optimizer.zero_grad()
        pred = model(xb)

        if "classification" in task_type:
            if yb.ndim == 1:  # multi-class
                loss = F.cross_entropy(pred, yb.long())
            else:  # multi-label
                loss = cls_criterion(pred.to(torch.float32), yb.to(torch.float32))
                # mask = ~torch.isnan(yb)
                # loss_mat = F.binary_cross_entropy_with_logits(
                #     pred.to(torch.float32), yb.to(torch.float32), reduction="none"
                # )
                # loss = (loss_mat * mask.float()).sum() / mask.float().sum()
        else:  # regression
            mask = ~torch.isnan(yb)
            loss_mat = F.mse_loss(pred, yb, reduction="none")
            loss = (loss_mat * mask.float()).sum() / mask.float().sum()

        loss.backward()
        optimizer.step()
def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, (xb, yb) in enumerate(tqdm(loader, desc="Iteration")):
        xb, yb = xb.to(device), yb.to(device)

        # squeeze if extra dimension
        if yb.ndim == 3 and yb.size(1) == 1:
            yb = yb.squeeze(1)

        with torch.no_grad():
            pred = model(xb)

        y_true.append(yb.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--c_channels', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molclintox",
                        help='dataset name (default: ogbg-molhiv)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset)
    print(dataset[0])

    list_hks, thres_hks,label = get_thresh_hks(dataset, 10, 0.1)
    list_deg, thres_deg = get_thresh(dataset, 10)
    y = torch.tensor(np.array(label), dtype=torch.float)
    graph_features = []
    for graph_id in tqdm(range(len(dataset))):
        b0, b1, node, edge = Topo_Fe_TimeSeries_MP(dataset[graph_id], list_deg[graph_id], list_hks[graph_id],
                                                   thres_deg, thres_hks)
        graph_features.append(torch.stack([b0, b1, node, edge], dim=0))
    MP_tensor = torch.stack(graph_features)
    y = torch.tensor(np.array(label), dtype=torch.float)
    X = MP_tensor
    print(MP_tensor)
    print(label)

    split_idx = dataset.get_idx_split()
    X_train, X_val,X_test = X[split_idx["train"]], X[split_idx["valid"]],X[split_idx["test"]]
    y_train, y_val,y_test = y[split_idx["train"]], y[split_idx["valid"]],y[split_idx["test"]]

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    valid_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    model = CNNTransformer(num_classes=dataset.num_tasks, cnn_channels=args.c_channels, d_model=args.d_model, drop_out=0.0, nhead=4,
                           num_layers=2)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, dataset.task_type)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)


if __name__ == "__main__":
    main()