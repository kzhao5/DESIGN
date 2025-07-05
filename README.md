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
