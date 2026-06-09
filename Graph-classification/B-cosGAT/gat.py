"""
full_gat_bcos_experiment.py
Runs BOTH:
  1) Standard GAT baseline with standard ELU activation
  2) BCos-GAT without ReLU/ELU activation, following the paper setting

Reports mean ± std accuracy and loss for ALL models.

Datasets: MUTAG, PROTEINS, or all
"""

import argparse
import random
import math
import os
import json
from statistics import mean, stdev

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_dataset(name: str, root: str = "data/TUDataset"):
    dataset = TUDataset(root=root, name=name)

    for data in dataset:
        if getattr(data, "x", None) is None:
            # Create degree feature
            deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.float)
            data.x = deg.view(-1, 1)
        else:
            if data.x.dim() == 1:
                data.x = data.x.view(-1, 1)

    return dataset


# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------
class BCosTemp(nn.Module):
    def __init__(self, B=1.2, temp=1.8, eps=1e-6, learn_B=False):
        super().__init__()
        self.temp = temp
        self.eps = eps

        if learn_B:
            raw_init = math.log(math.exp(B) - 1)
            self.raw_B = nn.Parameter(torch.tensor(raw_init))
        else:
            self.register_buffer("fixed_B", torch.tensor(float(B)))
            self.raw_B = None

    def B_value(self):
        if self.raw_B is None:
            return self.fixed_B
        return F.softplus(self.raw_B)

    def forward(self, x):
        B = self.B_value()
        norm = torch.sqrt((x * x).sum(-1, keepdim=True) + self.eps)
        x_norm = x / (norm + self.eps)
        x_scaled = self.temp * x_norm
        return x_scaled * (torch.abs(x_scaled) + self.eps).pow(B - 1)


class GATGraph(nn.Module):
    def __init__(self, in_channels, hidden=64, out_channels=2, heads=4, dropout=0.35):
        super().__init__()
        self.drop = dropout

        self.g1 = GATConv(in_channels, hidden, heads=heads, concat=True)
        self.n1 = nn.LayerNorm(hidden * heads)
        self.g2 = GATConv(hidden * heads, hidden, heads=1, concat=False)
        self.n2 = nn.LayerNorm(hidden)

        self.fc = nn.Linear(hidden, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.elu(self.n1(self.g1(x, edge_index)))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.elu(self.n2(self.g2(x, edge_index)))

        g = global_mean_pool(x, batch)
        return self.fc(g)


class BCosGATGraph(nn.Module):
    def __init__(self, in_channels, hidden=64, out_channels=2,
                 heads=4, dropout=0.35, B=1.2, learn_B=False):
        super().__init__()
        self.drop = dropout
        self.bcos = BCosTemp(B=B, temp=1.8, learn_B=learn_B)

        self.g1 = GATConv(in_channels, hidden, heads=heads, concat=True)
        self.n1 = nn.LayerNorm(hidden * heads)
        self.g2 = GATConv(hidden * heads, hidden, heads=1, concat=False)
        self.n2 = nn.LayerNorm(hidden)

        self.fc = nn.utils.weight_norm(nn.Linear(hidden, out_channels))

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=self.drop, training=self.training)

        # Paper setting: no ReLU/ELU activation is used in the B-cos model.
        x = self.n1(self.g1(x, edge_index))

        x = F.dropout(x, p=self.drop, training=self.training)

        # Paper setting: no ReLU/ELU activation is used in the B-cos model.
        h = self.n2(self.g2(x, edge_index))

        h_b = 0.7 * h + 0.3 * self.bcos(h)   # B-cos blend

        g = global_mean_pool(h_b, batch)
        return self.fc(g)


# -------------------------------------------------------------------
# Train / Eval Helper
# -------------------------------------------------------------------
def train_one_epoch(model, loader, opt, device):
    model.train()
    total = 0.0

    for data in loader:
        data = data.to(device)
        opt.zero_grad()

        logits = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(logits, data.y)

        loss.backward()
        opt.step()

        total += float(loss.item()) * data.num_graphs

    return total / len(loader.dataset)


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    losses = 0.0

    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(logits, data.y)

        preds = logits.argmax(-1)
        correct += int((preds == data.y).sum())
        total += data.num_graphs
        losses += float(loss.item()) * data.num_graphs

    return correct / total, losses / len(loader.dataset)


# -------------------------------------------------------------------
# K-FOLD RUNNER
# -------------------------------------------------------------------
def run_kfold(dataset, model_name, model_fn, model_kwargs,
              folds=10, epochs=200, batch_size=32,
              lr=0.005, wd=5e-4, device="cpu"):

    print(f"\n\n============ Running {model_name} ============")

    y = [int(d.y.item()) for d in dataset]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    test_scores = []
    val_scores = []
    losses = []

    fold = 0
    for train_idx, test_idx in skf.split(range(len(dataset)), y):
        fold += 1

        train_size = int(0.8 * len(train_idx))
        tr = train_idx[:train_size]
        va = train_idx[train_size:]

        train_set = [dataset[i] for i in tr]
        val_set = [dataset[i] for i in va]
        test_set = [dataset[i] for i in test_idx]

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)
        test_loader = DataLoader(test_set, batch_size=batch_size)

        in_ch = dataset[0].x.size(1)
        out_ch = dataset.num_classes

        model = model_fn(in_channels=in_ch, out_channels=out_ch, **model_kwargs).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        best_val = -1
        best_test = 0
        best_loss = None

        for ep in range(epochs):
            train_one_epoch(model, train_loader, opt, device)
            val_acc, val_loss = eval_model(model, val_loader, device)
            test_acc, _ = eval_model(model, test_loader, device)

            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc
                best_loss = val_loss

        print(f"Fold {fold:02d} | Val={best_val:.4f} | Test={best_test:.4f}")

        val_scores.append(best_val)
        test_scores.append(best_test)
        losses.append(best_loss)

    print("\n--- FINAL SUMMARY ---")
    print(f"{model_name}:")
    print(f"  Test   mean={mean(test_scores):.4f} ± {stdev(test_scores):.4f}")
    print(f"  Val    mean={mean(val_scores):.4f} ± {stdev(val_scores):.4f}")
    print(f"  Loss   mean={mean(losses):.4f} ± {stdev(losses):.4f}")

    return {
        "test": test_scores,
        "val": val_scores,
        "loss": losses
    }


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["MUTAG", "PROTEINS"],
        help="Dataset to run: MUTAG, PROTEINS"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--b_values", nargs="+", type=float,
                        default=[1.0, 1.2, 1.5, 2.0])
    parser.add_argument("--learn_B", action="store_true")
    parser.add_argument("--save_dir", type=str, default="results")

    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    set_seed(42)

    datasets_to_run = ["MUTAG", "PROTEINS"] if args.dataset == "all" else [args.dataset]
    os.makedirs(args.save_dir, exist_ok=True)

    all_results = {}

    for dataset_name in datasets_to_run:
        print(f"\n\n##################################################")
        print(f"Running dataset: {dataset_name}")
        print(f"##################################################")

        dataset = prepare_dataset(dataset_name)

        dataset_results = {}

        # ------------------------------
        # 1) RUN STANDARD GAT
        # ------------------------------
        gat_results = run_kfold(
            dataset,
            model_name=f"Standard GAT ({dataset_name})",
            model_fn=GATGraph,
            model_kwargs=dict(
                hidden=args.hidden,
                heads=args.heads,
                dropout=args.dropout
            ),
            folds=args.folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device
        )
        dataset_results["Standard GAT"] = gat_results

        # ------------------------------
        # 2) RUN BCOS-GAT FOR ALL B
        # ------------------------------
        for B in args.b_values:
            bcos_results = run_kfold(
                dataset,
                model_name=f"BCos-GAT ({dataset_name}, B={B}, learn_B={args.learn_B})",
                model_fn=BCosGATGraph,
                model_kwargs=dict(
                    hidden=args.hidden,
                    heads=args.heads,
                    dropout=args.dropout,
                    B=B,
                    learn_B=args.learn_B
                ),
                folds=args.folds,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device
            )
            dataset_results[f"BCos-GAT_B={B}"] = bcos_results

        all_results[dataset_name] = dataset_results

        dataset_outfile = os.path.join(args.save_dir, f"{dataset_name}_gat_bcos_graph_results.json")
        with open(dataset_outfile, "w") as f:
            json.dump(dataset_results, f, indent=2)
        print(f"Saved {dataset_name} summary to {dataset_outfile}")

    if len(datasets_to_run) > 1:
        combined_outfile = os.path.join(args.save_dir, "all_gat_bcos_graph_results.json")
        with open(combined_outfile, "w") as f:
            json.dump(all_results, f, indent=2)
        print("\n=== ALL DATASETS COMPLETED ===")
        print(f"Saved combined summary to {combined_outfile}")

