# Bcos-GNN
# Inherently Interpretable Graph Neural Networks via B-cos Alignment

Accepted at the **28th International Conference on Pattern Recognition (ICPR 2026)**

This repository contains the implementation of the paper:

**Inherently Interpretable Graph Neural Networks via B-cos Alignment**

---

## Repository Structure

```text
bcos-gnn/
├── README.md
├── requirements.txt
├── Datasets/
├── Node-classification/
│   ├── B-cosGCN/
│   │   ├── gcn-node.py
│   │   ├── hetro-gcn-node.py
│   │  
│   ├── B-cosGAT/
│   │   ├── gat-node.py
│   │   └── hetro-gat-node.py
│   ├── B-cosGraphSAGE/
│   │   ├── graphsage-node.py
│   │   ├── hetro-graphsage-node.py
│   │   
│   └── AnalysisB-cosGCN/
│       ├── B-cos-GCN-cora-multi-B-val.py
│       ├── B-cos-GCN-merged-dataset.py
│       └── bcos-gcn-node-anal6.py
├── Graph-classification/
│   ├── B-cosGCN/
│   │   ├── gcn.py
│   │   └── bcos-gcn-Mutag-anal6.py
│   ├── B-cosGAT/
│   │   └── gat.py
│   └── B-cosGraphSAGE/
│       └── graphsage.py
└── Images/
```

---

## Tested Environment

The experiments were tested using the following environment:

```text
Conda environment name: pyg
Python 3.10.19
torch==2.5.1
CUDA version: 12.1
CUDA available: True
GPU: NVIDIA GeForce RTX 2080 Ti
torch-geometric==2.7.0
numpy==2.2.6
scipy==1.15.2
scikit-learn==1.7.2
networkx==3.4.2
matplotlib==3.10.8
pandas==2.3.3
tqdm==4.67.1
torch-scatter==2.1.2+pt25cu121
torch-sparse==0.6.18+pt25cu121
torch-cluster==1.6.3+pt25cu121
torch-spline-conv==1.2.2+pt25cu121
```

---

## Installation

Create and activate a fresh conda environment:

```bash
conda create -n pyg python=3.10
conda activate pyg
```

Install PyTorch:

```bash
pip install torch==2.5.1
```

Install PyTorch Geometric compiled dependencies for PyTorch 2.5.1 and CUDA 12.1:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

Install the remaining dependencies:

```bash
pip install torch-geometric==2.7.0 numpy==2.2.6 scipy==1.15.2 scikit-learn==1.7.2 networkx==3.4.2 matplotlib==3.10.8 pandas==2.3.3 tqdm==4.67.1
```

Alternatively, install using:

```bash
pip install -r requirements.txt
```

---

## Environment Verification

After installation, verify the environment using:

```bash
python --version && \\
python -c "import torch; print('torch=='+torch.__version__)" && \\
python -c "import torch; print('CUDA version:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')" && \\
python -c "import torch_geometric; print('torch-geometric=='+torch_geometric.__version__)" && \\
python -c "import numpy; print('numpy=='+numpy.__version__)" && \\
python -c "import scipy; print('scipy=='+scipy.__version__)" && \\
python -c "import sklearn; print('scikit-learn=='+sklearn.__version__)" && \\
python -c "import networkx; print('networkx=='+networkx.__version__)" && \\
python -c "import matplotlib; print('matplotlib=='+matplotlib.__version__)" && \\
python -c "import pandas; print('pandas=='+pandas.__version__)" && \\
python -c "import tqdm; print('tqdm=='+tqdm.__version__)"
```

Optional PyTorch Geometric dependency check:

```bash
python -c "import torch_scatter; print('torch-scatter=='+torch_scatter.__version__)" && \\
python -c "import torch_sparse; print('torch-sparse=='+torch_sparse.__version__)" && \\
python -c "import torch_cluster; print('torch-cluster=='+torch_cluster.__version__)" && \\
python -c "import torch_spline_conv; print('torch-spline-conv=='+torch_spline_conv.__version__)"
```

---

## Datasets

The experiments use only publicly available datasets.

### Node Classification

```text
Cora
CiteSeer
PubMed
Texas
Cornell
```

### Graph Classification

```text
MUTAG
PROTEINS
```

The datasets are publicly available and can be downloaded automatically through PyTorch Geometric when the scripts are executed. No private dataset is used.

No pretrained model weights are required. All models are trained from scratch.

---

## Running Node Classification Experiments

### B-cos GCN

```bash
cd "Node-classification/B-cosGCN"
python gcn-node.py
```

For heterophilic datasets:

```bash
python hetro-gcn-node.py
```

### B-cos GraphSAGE

```bash
cd "Node-classification/B-cosGraphSAGE"
python graphsage-node.py
```

For heterophilic datasets:

```bash
python hetro-graphsage-node.py
```

### B-cos GAT

```bash
cd "Node-classification/B-cosGAT"
python gat-node.py
```

For heterophilic datasets:

```bash
python hetro-gat-node.py
```

---

## Running Graph Classification Experiments

### B-cos GCN

```bash
cd "Graph-classification/B-cosGCN"
python gcn.py
```

For MUTAG fidelity analysis:

```bash
python bcos-gcn-Mutag-anal6.py
```

### B-cos GraphSAGE

```bash
cd "Graph-classification/B-cosGraphSAGE"
python graphsage.py
```

### B-cos GAT

```bash
cd "Graph-classification/B-cosGAT"
python gat.py
```

---

## Fidelity-Sparsity Analysis

To reproduce the fidelity-sparsity analysis:

```bash
cd "Node-classification/AnalysisB-cosGCN"
python B-cos-GCN-cora-multi-B-val.py
```

For merged dataset analysis:

```bash
python B-cos-GCN-merged-dataset.py
```

The generated plots correspond to the figures stored in the `Images/` directory.

---

## Expected Results

Small numerical variations may occur due to random initialization, GPU nondeterminism, dataset splits, and library versions.

### Node Classification Results

| Model | Cora | CiteSeer | PubMed | Texas | Cornell |
|---|---:|---:|---:|---:|---:|
| B-cos GCN | 0.81 | 0.70 | 0.79 | 0.69 | 0.59 |
| B-cos GraphSAGE | 0.80 | 0.70 | 0.76 | 0.86 | 0.81 |
| B-cos GAT | 0.77 | 0.64 | 0.75 | 0.66 | 0.56 |

### Graph Classification Results

| Model | MUTAG | PROTEINS |
|---|---:|---:|
| B-cos GCN | 0.73 | 0.72 |
| B-cos GraphSAGE | 0.75 | 0.70 |
| B-cos GAT | 0.75 | 0.70 |

---

## Reproducibility Notes

The B-cos variants in this repository follow the paper setting, where ReLU/ELU activations are not used in the B-cos model path. 
Standard baseline models retain their usual nonlinear activations.

Small numerical variations from the reported paper results may occur due to random initialization, dataset splits, GPU/CUDA nondeterminism, and library-version differences.
In our reproduced runs, the results are generally consistent with the reported values, with typical deviations within approximately ±1 percentage point.

The B-cos exponent values are selected from:

```text
B ∈ {1.0, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0}
```

Results are reported as mean and standard deviation across multiple random initializations or cross-validation folds, as described in the paper.

---

## License

This code is released for research and reproducibility purposes. Please see the [LICENSE](LICENSE) file.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{shruti2026interpretable,
  title     = {Inherently Interpretable Graph Neural Networks via B-cos Alignment},
  author    = {Pandey, Shruti and Mishra, Subhankar},
  booktitle = {Proceedings of the 28th International Conference on Pattern Recognition (ICPR)},
  year      = {2026},
  month     = {August},
  note      = {To appear}
}
```

