<div align="center">

<h1>[NeurIPS 2025] DESIGN: Encrypte<strong>D</strong> GNN Inference via Server‑Side Input Graph Pruning</h1>

<div align="center">
  <a href="https://opensource.org/license/mit-0">
    <img alt="MIT‑0" src="https://img.shields.io/badge/License-MIT‑0-4E94CE.svg">
  </a>
  <a href="">
    <img src="https://img.shields.io/badge/Paper-ArXiv-darkred.svg" alt="Paper">
  </a>
  <a href="https://github.com/rai-lab-encrypted-gnn/design">
    <img src="https://img.shields.io/badge/Project-Page-924E7D.svg" alt="Project">
  </a>
</div>
</div>

> **DESIGN: EncrypteD GNN Inference via sErver‑Side Input Graph pruNing**  
> *Anonymous Authors* (NeurIPS 2025, paper ID XXXX)  

## ✨ Abstract

Graph Neural Networks (GNNs) executed under Fully Homomorphic Encryption (FHE) allow third‑party servers to perform inference on encrypted graphs, but existing methods are far too slow for real‑time use. **DESIGN** achieves order‑of‑magnitude speed‑ups by  
1. computing encrypted node‑degree statistics,  
2. homomorphically *partitioning* nodes into importance levels,  
3. **pruning** low‑importance nodes/edges, and  
4. assigning **adaptive polynomial activations** whose degree decreases with node importance.  
Extensive experiments across five benchmarks demonstrate substantial latency reduction with negligible accuracy loss relative to plaintext inference. :contentReference[oaicite:0]{index=0}

---

## 🖥️ Environment Setup

| Component | Recommended setting |
|-----------|--------------------|
| CPU | ≥ 24 cores (AVX2 or AVX‑512) |
| GPU | *Optional* – used **only** for plaintext model training |
| OS | Linux (Ubuntu 20.04 LTS tested) |
| FHE libs | **Microsoft SEAL 4.2** (CKKS) & **OpenFHE 1.2.3** |

### Quick start

```bash
git clone https://github.com/rai-lab-encrypted-gnn/design.git
cd design

# 1. Python dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # <–– see file below

# 2. (Optional) build SEAL/OpenFHE from source if no wheel available
#    Instructions: docs/build_seal.md  and docs/build_openfhe.md


## Prerequisites

1.  **Python Environment:** Ensure you have Python 3.8+ installed.
2.  **Dependencies:** Install the required Python packages:
    ```bash
    pip install torch torchvision torchaudio torch_geometric
    pip install PyYAML # For SEAL if building from source, or if SEAL python wheel needs it
    pip install numpy scikit-learn ogb
    # Install Microsoft SEAL Python wrapper
    pip install seal
    # Install OpenFHE Python wrapper (if you have it set up)
    # pip install openfhe
    ```
3.  **Pretrained GCN Models:**
    *   Train plaintext GCN models using `train_plaintext_model.py`.
    *   Ensure the models are saved in the `./models/` directory (e.g., `./models/Cora.pt`, `./models/CiteSeer.pt`, etc.).
    ```bash
    python train_plaintext_model.py
    ```
4.  **Datasets:** The scripts will automatically download datasets (Cora, CiteSeer, PubMed, Yelp, ogbn-proteins, KarateClub) to a `./data/` directory if they are not already present.

## Directory Structure

Ensure your project has a structure similar to this:

```
.
├── design_model.py         # Your proposed method
├── seal_model.py           # Baseline SEAL implementation
├── openfhe_model.py        # Baseline OpenFHE implementation (if used)
├── gcn_model.py            # PyTorch GCN model definition
├── train.py                # Training script components
├── train_plaintext_model.py # Script to train and save models
├── utils.py                # Utility functions
├── models/                 # Directory for saved plaintext models
│   ├── Cora.pt
│   └── ...
├── data/                   # Directory for datasets
└── README.md               # This file
```

## Running Experiments

All results will be appended to CSV files: `seal_results.csv`, `openfhe_results.csv`, and `design_results.csv`.

### 1. Answering RQ1 & RQ2: Latency Reduction and Accuracy Impact

These questions involve comparing the full DESIGN framework against baselines (plaintext, SEAL FHE, OpenFHE FHE).

**A. Run Plaintext Baseline (Reference):**
The plaintext accuracy is reported by `design_model.py` and other FHE scripts as a reference. No separate run is needed if you are running the FHE scripts.

**B. Run Baseline FHE Implementations:**

*   **SEAL Baseline:**
    For each dataset (e.g., Cora, CiteSeer, PubMed, Yelp, proteins, Karate):
    ```bash
    python seal_model.py Cora
    python seal_model.py CiteSeer
    # ... and so on for other datasets
    ```
    Results will be logged to `seal_results.csv`.

*   **OpenFHE Baseline (if applicable):**
    For each dataset:
    ```bash
    python openfhe_model.py Cora
    python openfhe_model.py CiteSeer
    # ... and so on for other datasets
    ```
    Results will be logged to `openfhe_results.csv`.

**C. Run DESIGN Framework (Full Method):**
This run enables both pruning and adaptive activation.
For each dataset:
```bash
python design_model.py Cora prune:on adaptive_act:on
python design_model.py CiteSeer prune:on adaptive_act:on
python design_model.py PubMed prune:on adaptive_act:on
python design_model.py Yelp prune:on adaptive_act:on
python design_model.py proteins prune:on adaptive_act:on
python design_model.py Karate prune:on adaptive_act:on
```
Results will be logged to `design_results.csv`.

**Analysis for RQ1 & RQ2:**
Compare the `FHEAcc` and `Latency` columns from `design_results.csv` (with prune:on, adaptive_act:on) against `seal_results.csv`, `openfhe_results.csv`, and the `PlainAcc` reported in the CSVs.

### 2. Answering RQ3: Ablation Study on Framework Components

This involves running the DESIGN framework with different components enabled/disabled.

**A. DESIGN with Pruning Only (Adaptive Activation OFF):**
For each dataset:
```bash
python design_model.py Cora prune:on adaptive_act:off
python design_model.py CiteSeer prune:on adaptive_act:off
# ... and so on for other datasets
```

**B. DESIGN with Adaptive Activation Only (Pruning OFF):**
For each dataset:
```bash
python design_model.py Cora prune:off adaptive_act:on
python design_model.py CiteSeer prune:off adaptive_act:on
# ... and so on for other datasets
```

**C. DESIGN with Both Pruning and Adaptive Activation OFF (Optional):**
This configuration effectively runs a GCN under FHE with standard square activation, similar to the baselines but within the DESIGN script's structure.
For each dataset:
```bash
python design_model.py Cora prune:off adaptive_act:off
python design_model.py CiteSeer prune:off adaptive_act:off
# ... and so on for other datasets
```

**Analysis for RQ3:**
Examine `design_results.csv`. Compare rows with different `Pruning` and `AdaptiveAct` flags for the same dataset to understand the individual and combined contributions of these components to accuracy and latency.

### 3. Answering RQ4: Sensitivity Analysis of Hyperparameters

This requires modifying the `design_model.py` script directly to change hyperparameter values and then re-running the experiments. Always run with both pruning and adaptive activation ON for these tests, unless specifically testing their interaction with hyperparameter changes.

**A. Impact of Pruning Ratio (via Pruning Thresholds \(T_m\)):**

1.  **Modify `design_model.py`:**
    Locate the `pruning_thresholds` variable within the `if __name__ == '__main__':` block.
    ```python
    # Example in design_model.py
    # ...
    # pruning_thresholds = [5.0, 2.0] # Original example
    # To test different thresholds for a dataset (e.g., Cora):
    if dataset_name == "Cora":
        # Test 1: More aggressive pruning (higher T_m, or fewer levels with higher thresholds)
        # pruning_thresholds = [7.0, 4.0]
        # Test 2: Less aggressive pruning
        # pruning_thresholds = [3.0, 1.0]
        # Test 3: Single pruning threshold (T_m)
        pruning_thresholds = [3.0] # Nodes with degree < 3 pruned. One level for retained.
                                   # Ensure adaptive_poly_coeffs matches length if adaptive_act is on.
    # ...
    ```
    If you use a single threshold `[T_m]`, and adaptive activation is ON, ensure `adaptive_poly_coeffs` also has only one polynomial configuration. For example:
    ```python
    if dataset_name == "Cora":
        pruning_thresholds = [3.0]
        adaptive_poly_coeffs = [
            [0, 0, 1.0],  # x^2 for all retained nodes (score >= 3.0)
        ]
    ```

2.  **Run the script for the chosen dataset(s):**
    ```bash
    python design_model.py Cora prune:on adaptive_act:on
    ```
    Repeat for different threshold settings.

**B. Impact of Polynomial Degrees for Adaptive Activation:**

1.  **Modify `design_model.py`:**
    Locate the `adaptive_poly_coeffs` variable within the `if __name__ == '__main__':` block.
    The number of polynomial configurations in `adaptive_poly_coeffs` must match the number of thresholds in `pruning_thresholds`.
    ```python
    # Example in design_model.py
    # ...
    # Original example for pruning_thresholds = [5.0, 2.0]
    # adaptive_poly_coeffs = [
    #     [0, 0, 1.0],  # P_d1: x^2 for level 1 (score >= 5.0)
    #     [0, 1.0],     # P_d2: x   for level 2 (2.0 <= score < 5.0)
    # ]

    # To test different polynomial sets for a dataset (e.g., Cora),
    # assuming pruning_thresholds = [5.0, 2.0] is kept:
    if dataset_name == "Cora":
        # Test 1: Higher fidelity polynomials (example: PSet1 from paper (7,5,3))
        # This would require implementing degree 7, 5, 3 polynomial evaluation.
        # For simplicity, let's use (3,2) for (P_d1, P_d2)
        # P_d1 = x^3 (coeffs: [0,0,0,1]), P_d2 = x^2 (coeffs: [0,0,1])
        # adaptive_poly_coeffs = [
        #    [0,0,0,1.0], # x^3
        #    [0,0,1.0]    # x^2
        # ]

        # Test 2: Lower fidelity polynomials (example: PSet3 from paper (3,2,1))
        # P_d1 = x (coeffs: [0,1]), P_d2 = 1 (coeffs: [1.0]) (constant activation)
        adaptive_poly_coeffs = [
           [0, 1.0],    # x
           [1.0]        # 1 (constant)
        ]
    # ...
    ```
    **Note:** The `_eval_poly` function in `design_model.py` uses Horner's method. Coefficients are `[c0, c1, c2, ..., cn]` for \(P(x) = c_0 + c_1x + c_2x^2 + \dots + c_nx^n\).

2.  **Run the script for the chosen dataset(s):**
    ```bash
    python design_model.py Cora prune:on adaptive_act:on
    ```
    Repeat for different polynomial degree settings. Remember to keep the `pruning_thresholds` consistent for a fair comparison of polynomial sets, or vary them systematically.

**Analysis for RQ4:**
Collect data from `design_results.csv` for runs with different hyperparameter settings. Plot accuracy and latency against pruning ratios (derived from thresholds) or polynomial degree sets to observe trends.

## Output Files

*   `seal_results.csv`: Results from the SEAL baseline.
    Columns: `Dataset,FHEAcc,PlainAcc,Latency,Rot,PMult,CMult,Add`
*   `openfhe_results.csv`: Results from the OpenFHE baseline (if used).
    Columns: `Dataset,FHEAcc,PlainAcc,Latency,Rot,PMult,CMult,Add`
*   `design_results.csv`: Results from the DESIGN framework.
    Columns: `Dataset,Pruning,AdaptiveAct,PlainAcc,FHEAcc,Latency,Rot,PMult,CMult,Add,PolyEval,MaskGen`
    *   `Pruning`: `True` if pruning was enabled, `False` otherwise.
    *   `AdaptiveAct`: `True` if adaptive activation was enabled, `False` otherwise.
    *   Other columns are metrics or operation counts.

## General Notes

*   **Dataset Names:** Use one of `Cora`, `CiteSeer`, `PubMed`, `Yelp`, `proteins`, `Karate`.
*   **FHE Parameters:** The FHE parameters (`poly_modulus_degree`, `coeff_modulus`, `scale`) in `design_model.py` (and baseline scripts) might need adjustment if you encounter excessive noise (leading to incorrect results) or performance issues, especially when using higher-degree polynomials or deeper computational graphs resulting from certain hyperparameter choices.
*   **Experiment Duration:** FHE computations are slow. Experiments, especially on larger datasets or with complex settings, can take a significant amount of time. Start with smaller datasets or simpler configurations.
*   **Consistency:** For fair comparisons, ensure that the underlying GCN model (`./models/<dataset_name>.pt`) is the same across all relevant FHE runs for a given dataset.
```
