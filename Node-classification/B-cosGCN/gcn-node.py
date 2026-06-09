#!/usr/bin/env python3
# ===========================================================
# Standard GCN and BCos-GCN on Planetoid Datasets
# Datasets: Cora, CiteSeer, PubMed, or all
# Node classification with multi-seed evaluation
#
# StandardGCN uses ReLU as the standard baseline.
# BCosGCN follows the paper setting without ReLU/ELU activation.
# ===========================================================

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree


# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_dataset_name(name: str) -> str:
    mapping = {
        "cora": "Cora",
        "citeseer": "CiteSeer",
        "pubmed": "PubMed",
        "all": "all",
    }
    key = name.strip().lower()
    if key not in mapping:
        raise ValueError("Unknown dataset. Use one of: Cora, CiteSeer, PubMed, all.")
    return mapping[key]


def normalize_adj_edge_weight(edge_index, num_nodes, dtype=torch.float):
    edge_index_with_self, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index_with_self

    deg = degree(col, num_nodes=num_nodes, dtype=dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return edge_index_with_self, edge_weight


# -----------------------------------------------------------
# B-cos layer
# -----------------------------------------------------------
class BCosGCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, B: float = 2.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.B = float(B)
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * (1.0 / np.sqrt(in_features))
        )

    def forward(self, z, edge_index=None, edge_weight=None):
        lin = torch.matmul(z, self.weight.t())

        z_norm = F.normalize(z, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)

        cos = torch.matmul(z_norm, w_norm.t())
        scale = cos.abs().pow(self.B - 1.0)

        return lin * scale


# -----------------------------------------------------------
# BCos-GCN model
# -----------------------------------------------------------
class BCosGCN(nn.Module):
    """BCos-GCN model following the paper setting without ReLU/ELU activation."""

    def __init__(self, in_channels, hidden, out_channels, B=2.0, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.layer1 = BCosGCNLayer(in_channels, hidden, B)
        self.layer2 = BCosGCNLayer(hidden, out_channels, B)

    def forward(self, x, edge_index, edge_weight):
        row, col = edge_index

        z = torch.zeros_like(x)
        z.index_add_(0, row, x[col] * edge_weight.unsqueeze(-1))

        h1 = self.layer1(z)

        # Paper setting: no ReLU/ELU activation is used in the B-cos model.
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        z2 = torch.zeros_like(h1)
        z2.index_add_(0, row, h1[col] * edge_weight.unsqueeze(-1))

        return self.layer2(z2)


# -----------------------------------------------------------
# Standard GCN baseline model
# -----------------------------------------------------------
class StandardGCN(nn.Module):
    """Standard GCN baseline with ReLU activation."""

    def __init__(self, in_channels, hidden, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden, bias=False)
        self.conv2 = GCNConv(hidden, out_channels, bias=False)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


# -----------------------------------------------------------
# Train / Eval routines
# -----------------------------------------------------------
def train_epoch(model, data, optimizer, device, edge_index, edge_weight):
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(device), edge_index.to(device), edge_weight.to(device))
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].to(device))
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, device, edge_index, edge_weight):
    model.eval()
    out = model(data.x.to(device), edge_index.to(device), edge_weight.to(device))
    preds = out.argmax(dim=1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = preds[mask].eq(data.y[mask].to(device)).sum().item()
        accs.append(correct / int(mask.sum().item()))
    return accs, out


# ===========================================================
# Multi-seed experiment for one model
# ===========================================================
def run_single_model(model_class, model_args, data, edge_index, edge_weight,
                     lr, weight_decay, epochs, device, seeds):

    val_list = []
    test_list = []
    loss_list = []

    for seed in seeds:
        set_seed(seed)

        model = model_class(*model_args).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        best_val = 0.0
        best_test = 0.0
        final_loss = 0.0

        for _ in range(epochs):
            final_loss = train_epoch(model, data, optimizer, device, edge_index, edge_weight)
            accs, _ = evaluate(model, data, device, edge_index, edge_weight)
            train_acc, val_acc, test_acc = accs

            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc

        val_list.append(best_val)
        test_list.append(best_test)
        loss_list.append(final_loss)

    return {
        "val_mean": float(np.mean(val_list)),
        "val_std": float(np.std(val_list)),
        "test_mean": float(np.mean(test_list)),
        "test_std": float(np.std(test_list)),
        "loss_mean": float(np.mean(loss_list)),
        "loss_std": float(np.std(loss_list)),
        "all_val_scores": [float(x) for x in val_list],
        "all_test_scores": [float(x) for x in test_list],
        "all_losses": [float(x) for x in loss_list],
    }


# ===========================================================
# Tuning Loop
# ===========================================================
def run_tuning(dataset_name, device, B_grid, hidden, lr, weight_decay, epochs, base_seed, save_dir):
    print(f"\n=== Dataset: {dataset_name} | device={device} ===")

    dataset = Planetoid(root=f"./data/{dataset_name}", name=dataset_name)
    data = dataset[0]

    edge_index, edge_weight = normalize_adj_edge_weight(data.edge_index, data.num_nodes)

    in_channels = dataset.num_node_features
    out_channels = dataset.num_classes

    # Multi-seed setup
    seeds = [base_seed + i for i in range(10)]

    # ------------------------------------------------------
    # Standard GCN
    # ------------------------------------------------------
    print("\nRunning Standard GCN across 10 seeds...")
    standard_results = run_single_model(
        StandardGCN,
        (in_channels, hidden, out_channels),
        data,
        edge_index,
        edge_weight,
        lr,
        weight_decay,
        epochs,
        device,
        seeds,
    )

    print("\n=== Standard GCN Results (mean ± std) ===")
    print(f"Val Acc : {standard_results['val_mean']:.4f} ± {standard_results['val_std']:.4f}")
    print(f"Test Acc: {standard_results['test_mean']:.4f} ± {standard_results['test_std']:.4f}")
    print(f"Loss    : {standard_results['loss_mean']:.4f} ± {standard_results['loss_std']:.4f}")

    # ------------------------------------------------------
    # BCos-GCN for each B
    # ------------------------------------------------------
    bcos_results = {}

    for B in B_grid:
        print(f"\nRunning BCos-GCN (B={B}) across 10 seeds...")
        res = run_single_model(
            BCosGCN,
            (in_channels, hidden, out_channels, B),
            data,
            edge_index,
            edge_weight,
            lr,
            weight_decay,
            epochs,
            device,
            seeds,
        )
        bcos_results[str(B)] = res

        print(f"\n=== BCos-GCN (B={B}) Results ===")
        print(f"Val Acc : {res['val_mean']:.4f} ± {res['val_std']:.4f}")
        print(f"Test Acc: {res['test_mean']:.4f} ± {res['test_std']:.4f}")
        print(f"Loss    : {res['loss_mean']:.4f} ± {res['loss_std']:.4f}")

    # Select best B by validation mean
    best_B = max(bcos_results.keys(), key=lambda b: bcos_results[b]["val_mean"])
    best_bcos = bcos_results[best_B]

    print("\n================ FINAL SUMMARY ================")
    print(f"Dataset: {dataset_name}")
    print(
        f"Standard GCN Test: {standard_results['test_mean']:.4f} ± "
        f"{standard_results['test_std']:.4f}"
    )
    print(
        f"Best BCos-GCN B={best_B} Test: {best_bcos['test_mean']:.4f} ± "
        f"{best_bcos['test_std']:.4f}"
    )

    result = {
        "dataset": dataset_name,
        "standard_gcn": standard_results,
        "bcos_gcn_grid": bcos_results,
        "best_bcos_gcn": {
            "B": float(best_B),
            **best_bcos,
        },
    }

    os.makedirs(save_dir, exist_ok=True)
    outfile = os.path.join(save_dir, f"{dataset_name}_bcos_gcn_node_results.json")
    with open(outfile, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Summary saved to {outfile}")

    return result


# ===========================================================
# CLI
# ===========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["Cora", "CiteSeer", "PubMed", "all"],
        help="Dataset to run: Cora, CiteSeer, PubMed, or all",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="results")
    args = parser.parse_args()

    B_grid = [1.0, 1.5, 2.0, 2.5, 3.0]

    selected = canonical_dataset_name(args.dataset)
    datasets_to_run = ["Cora", "CiteSeer", "PubMed"] if selected == "all" else [selected]

    all_results = {}
    for dataset_name in datasets_to_run:
        all_results[dataset_name] = run_tuning(
            dataset_name,
            device=args.device,
            B_grid=B_grid,
            hidden=args.hidden,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            base_seed=args.seed,
            save_dir=args.save_dir,
        )

    if len(datasets_to_run) > 1:
        os.makedirs(args.save_dir, exist_ok=True)
        combined_file = os.path.join(args.save_dir, "all_bcos_gcn_node_results.json")
        with open(combined_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print("\n=== ALL DATASETS COMPLETED ===")
        print(f"Combined summary saved to {combined_file}")

