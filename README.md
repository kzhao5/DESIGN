<div align="center">

<h1>DESIGN: Encrypte<strong>D</strong> GNN Inference via Server-Side Input Graph Pruning</h1>

</div>

> **DESIGN: EncrypteD GNN Inference via Server-Side Input Graph Pruning**

---

## ✨ Abstract
Graph Neural Networks (GNNs) under Fully Homomorphic Encryption (FHE) are extremely slow. **DESIGN** accelerates encrypted inference by 
1. encrypting node-degree statistics, 
2. partitioning nodes into importance levels, 
3. **pruning** low-importance nodes/edges, and 
4. using **adaptive polynomial activations** (lower degree for less-important nodes). 
Across five benchmarks DESIGN achieves up to **2 × latency reduction** with ≤ 2 pp accuracy loss vs. plaintext.

---

## 1 – Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python**  | ≥ 3.8 |
| **ML stack**| `torch`, `torch_geometric`, `numpy`, `scikit-learn`, `ogb` |
| **FHE libs**| `seal` (≥ 4.2) · (optional) `openfhe` (≥ 1.2) |
| **Helpers** | `PyYAML` (needed if compiling SEAL from source) |

```bash
# Obtain the source (anonymous package or local copy)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# If no wheel exists for SEAL/OpenFHE, follow docs/build_seal.md & docs/build_openfhe.md


 ---

 ## 2 – Directory Structure
 ```
 .
 ├── design_model.py            # DESIGN (SEAL) – pruning + adaptive act
 ├── seal_model.py              # baseline SEAL GCN
 ├── openfhe_model.py           # baseline OpenFHE GCN  (optional)
 ├── gcn_model.py               # 2-layer backbone
 ├── train.py                   # training helpers
 ├── train_plaintext_model.py   # pre-trains and saves *.pt
 ├── utils.py                   # data utilities
 ├── models/                    # pretrained plaintext GCNs (auto-generated)
 ├── data/                      # auto-downloaded datasets
 ├── results/                   # CSV logs (auto-generated)
 └── README.md                  # ← this file
 ```

 ## 3 – Running Experiments
 All commands append one line to `results/{seal,openfhe,design}_results.csv`.

 ### 3.1  Pre-train plaintext models (one-time)
 ```bash
 python train_plaintext_model.py   # models/<dataset>.pt are created
 ```
<details>
<summary>Sample Output</summary>
 
[INFO] Loading dataset: Cora...
[INFO] Training GCN model on Cora...
Epoch 100 | Train Loss: 0.1234 | Val Acc: 0.8050 | Test Acc: 0.8120
[INFO] Plaintext training complete. Best validation accuracy: 0.8150
[INFO] Saving trained model to models/Cora.pt
...
[INFO] All models trained and saved in ./models/

</details>

 ### 3.2  Baselines (SEAL / OpenFHE)  *RQ1 & RQ2 reference*
 ```bash
 # SEAL baseline
 python seal_model.py Cora
 python seal_model.py CiteSeer
 # ... (PubMed, Yelp, proteins, Karate)

 # OpenFHE baseline (optional)
 python openfhe_model.py Cora
 python openfhe_model.py CiteSeer
 # ...
 ```
 <details> <summary>Sample Output (for <code>seal_model.py Cora</code>)</summary>
[INFO] Running SEAL Baseline for dataset: Cora
[INFO] Loading plaintext model from models/Cora.pt and data...
[INFO] Initializing FHE context (SEAL)...
[INFO] Encrypting graph data... done (2.1s)
[INFO] Performing FHE inference... done (45.8s)
[INFO] Decrypting results... done (0.5s)
[RESULT] Plaintext Acc: 0.8120 | FHE Acc: 0.8115 | Latency: 45.8s
[INFO] Appending results to results/seal_results.csv
</details>

 ### 3.3  DESIGN – Full Framework  *RQ1 & RQ2 main*
 ```bash
 python design_model.py Cora     prune:on adaptive_act:on
 python design_model.py CiteSeer prune:on adaptive_act:on
 python design_model.py PubMed   prune:on adaptive_act:on
 python design_model.py Yelp     prune:on adaptive_act:on
 python design_model.py proteins prune:on adaptive_act:on
 python design_model.py Karate   prune:on adaptive_act:on
 ```

<details> <summary>Sample Output (for <code>design_model.py Cora prune:on adaptive_act:off</code>)</summary>
[INFO] Running DESIGN for dataset: Cora with Pruning=ON, AdaptiveAct=OFF
[INFO] Loading plaintext model from models/Cora.pt and data...
[INFO] Initializing FHE context (SEAL)...
[INFO] Generating importance masks... done (5.5s)
[INFO] Pruning graph and performing FHE inference with square activation... done (31.2s)
[INFO] Decrypting results... done (0.4s)
[RESULT] Plaintext Acc: 0.8120 | FHE Acc: 0.8095 | Latency: 36.7s
[INFO] Appending results to results/design_results.csv
</details>

 ### 3.4  Ablations  *RQ3*
 ```bash
 # pruning only
 python design_model.py Cora prune:on  adaptive_act:off
 # adaptive activation only
 python design_model.py Cora prune:off adaptive_act:on
 # neither component (square activation)
 python design_model.py Cora prune:off adaptive_act:off
 ```

<details> <summary>Sample Output (for <code>design_model.py Cora prune:on adaptive_act:off</code>)</summary>
[INFO] Running DESIGN for dataset: Cora with Pruning=ON, AdaptiveAct=OFF
[INFO] Loading plaintext model from models/Cora.pt and data...
[INFO] Initializing FHE context (SEAL)...
[INFO] Generating importance masks... done (5.5s)
[INFO] Pruning graph and performing FHE inference with square activation... done (31.2s)
[INFO] Decrypting results... done (0.4s)
[RESULT] Plaintext Acc: 0.8120 | FHE Acc: 0.8095 | Latency: 36.7s
[INFO] Appending results to results/design_results.csv
</details>

 ## 4 – Research Questions → Command Map
 | RQ  | Goal                                              | Command(s)                      |
 |-----|---------------------------------------------------|---------------------------------|
 | RQ1 | Latency reduction vs. SEAL/OpenFHE                | § 3.3                           |
 | RQ2 | Accuracy impact vs. plaintext & FHE baselines     | compare `FHEAcc` in CSVs        |
 | RQ3 | Component contribution                            | § 3.4                           |
 | RQ4 | Hyper-parameter sensitivity                       | edit `pruning_thresholds` & `adaptive_poly_coeffs` in `design_model.py`, then run § 3.3 |

 ## 5 – Hyper-parameter Tuning (RQ4)
 *Open **design_model.py** (bottom)*
 ```python
 pruning_thresholds = [5.0, 2.0]          # default two-level
 adaptive_poly_coeffs = [
     [0, 0, 1.0],  # x²
     [0, 1.0]      # x
 ]
 ```
 **Examples**
 ```python
 # single aggressive threshold
 pruning_thresholds = [3.0]
 adaptive_poly_coeffs = [[0, 0, 1.0]]      # x²

 # high-fidelity
 pruning_thresholds = [7.0, 4.0]
 adaptive_poly_coeffs = [
     [0,0,0,1.0],   # x³
     [0,0,1.0]      # x²
 ]
 ```
 Then rerun:
 ```bash
 python design_model.py Cora prune:on adaptive_act:on
 ```

 ## 6 – Output Files
 | File                                  | Columns                                                        |
 |---------------------------------------|----------------------------------------------------------------|
 | `seal_results.csv`, `openfhe_results.csv` | `Dataset,FHEAcc,PlainAcc,Latency,Rot,PMult,CMult,Add`          |
 | `design_results.csv`                  | `Dataset,Pruning,AdaptiveAct,PlainAcc,FHEAcc,Latency,Rot,PMult,CMult,Add,PolyEval,MaskGen` |

 ## 7 – General Notes
 * **Datasets:** Cora, CiteSeer, PubMed, Yelp, proteins, Karate.
 * **FHE parameters** (`poly_modulus_degree`, `coeff_modulus`, `scale`) may need tuning for deep polynomials.
 * **Runtime:** FHE is slow; start with Cora/Karate for sanity checks.
 * **Consistency:** reuse the same `.pt` for a given dataset across runs.
