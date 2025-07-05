import sys
import time
import math
import random
import numpy as np
import torch
from torch_geometric.datasets import Planetoid, KarateClub
from torch_geometric.transforms import RandomNodeSplit
from train import train_gcn, test
from utils import get_A_bar, get_subgraph, get_predictions, test_acc_calc, get_dataset

import seal
from seal import EncryptionParameters, scheme_type, SEALContext
from seal import KeyGenerator, CKKSEncoder, Encryptor, Evaluator, Decryptor
from seal import Plaintext, Ciphertext, RelinKeys, GaloisKeys
import math

# Parameters
poly_modulus_degree = 8192
coeff_modulus = seal.CoeffModulus.Create(poly_modulus_degree, [34, 30, 30, 30, 30, 30, 34])
scale = pow(2.0, 30)

# Setup SEAL context and keys
def setup_seal(num_nodes: int):
    parms = EncryptionParameters(scheme_type.ckks)
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(coeff_modulus)
    context = SEALContext(parms)

    keygen = KeyGenerator(context)
    public_key = keygen.create_public_key()
    secret_key = keygen.secret_key()
    relin_keys = keygen.create_relin_keys()
    galois_keys = keygen.create_galois_keys()

    encoder = CKKSEncoder(context)
    encryptor = Encryptor(context, public_key)
    evaluator = Evaluator(context)
    decryptor = Decryptor(context, secret_key)

    return {
        'context': context,
        'encoder': encoder,
        'encryptor': encryptor,
        'evaluator': evaluator,
        'decryptor': decryptor,
        'relin_keys': relin_keys,
        'galois_keys': galois_keys,
    }



# -----------------------------------------------------------------------------
# Operation Counters
# -----------------------------------------------------------------------------
op_counts = {"Rot": 0, "PMult": 0, "CMult": 0, "Add": 0}

# Utility to print counts
def print_op_counts():
    print("\n=== SEAL Homomorphic Op Counts ===")
    for k, v in op_counts.items():
        print(f"  {k:5s} : {v}")
    print("======================================\n")

# Encrypt GCN class
class EncGCN:
    def __init__(self, seal_ctx, torch_model):
        self.ctx = seal_ctx
        self.encoder = seal_ctx['encoder']
        self.encryptor = seal_ctx['encryptor']
        self.evaluator = seal_ctx['evaluator']
        self.relin_keys = seal_ctx['relin_keys']
        self.galois_keys = seal_ctx['galois_keys']
        self.scale = scale

        # Weight matrices as lists
        self.W1 = torch_model.conv1.lin.weight.T.detach().tolist()
        self.W2 = torch_model.conv2.lin.weight.T.detach().tolist()

    def encrypt_adj(self, adj_norm):
        adj_diags = []
        self.N = len(adj_norm)
        for d in range(self.N):
            diag = [adj_norm[i][(i + d) % self.N] for i in range(self.N)]
            pt = self.encoder.encode(diag, self.scale)
            ct = self.encryptor.encrypt(pt)
            adj_diags.append(ct)

        self.adj_diags = adj_diags

    def encrypt_features(self, feats):
        # feats: list of F lists of length N
        enc_feats = []
        for fv in feats:
            pt = self.encoder.encode(fv, self.scale)
            ct = self.encryptor.encrypt(pt)
            enc_feats.append(ct)
        return enc_feats

    def _aggregate(self, ct_feat):
        s = math.isqrt(self.N) + 1  # baby step size
        m = (self.N + s - 1) // s   # giant step count

        # Precompute baby steps (rotated inputs)
        baby_steps = {}
        for j in range(s):
            if j >= self.N:
                break
            rot = self.evaluator.rotate_vector(ct_feat, j, self.galois_keys)
            baby_steps[j] = rot
            op_counts["Rot"] += 1

        agg = None
        for i in range(m):
            # Inner sum for this giant step
            inner_sum = None
            for j in range(s):
                idx = i * s + j
                if idx >= self.N:
                    break
                rotated = baby_steps[j]
                self.evaluator.mod_switch_to_inplace(self.adj_diags[idx], rotated.parms_id())
                tmp = self.evaluator.multiply(rotated, self.adj_diags[idx])
                op_counts["CMult"] += 1
                self.evaluator.relinearize_inplace(tmp, self.relin_keys)
                self.evaluator.rescale_to_next_inplace(tmp)

                if inner_sum is None:
                    inner_sum = tmp
                else:
                    self.evaluator.mod_switch_to_inplace(tmp, inner_sum.parms_id())
                    self.evaluator.add_inplace(inner_sum, tmp)
                    op_counts["Add"] += 1

            # Rotate inner sum by i*s
            if i > 0:
                inner_sum = self.evaluator.rotate_vector(inner_sum, i * s, self.galois_keys)
                op_counts["Rot"] += 1

            if agg is None:
                agg = inner_sum
            else:
                self.evaluator.mod_switch_to_inplace(inner_sum, agg.parms_id())
                self.evaluator.add_inplace(agg, inner_sum)
                op_counts["Add"] += 1

        return agg

    def eval_linear_wsum(self, ctxts: list[Ciphertext], weights: list[float]) -> Ciphertext:
        """
        Homomorphically compute sum_i weights[i] * ctxts[i]
        by multiplying each ciphertext by a scalar-plaintext.
        """
        out_ct = None
        for ct_i, w in zip(ctxts, weights):
            # 1) encode the scalar weight
            pt_w = self.encoder.encode(float(w), self.scale)

            # 2) match levels: mod-switch plaintext to ciphertext's parms
            self.evaluator.mod_switch_to_inplace(pt_w, ct_i.parms_id())

            # 3) multiply plaintext*ct
            prod = self.evaluator.multiply_plain(ct_i, pt_w)
            op_counts["PMult"] += 1

            # 4) relinearize & rescale
            self.evaluator.relinearize_inplace(prod, self.relin_keys)
            self.evaluator.rescale_to_next_inplace(prod)

            # 5) accumulate
            if out_ct is None:
                out_ct = prod
            else:
                self.evaluator.mod_switch_to_inplace(prod, out_ct.parms_id())
                self.evaluator.add_inplace(out_ct, prod)
                op_counts["Add"] += 1

        return out_ct


    def forward(self, enc_feats):
        # Layer1: linear0
        # aggregate
        print("Layer 1 agg starting")
        enc_agg = [self._aggregate(ct) for ct in enc_feats]
        print("Layer 1 agg finished")
        # linear
        print("Layer 1 liner started")
        F = len(self.W1)
        H = len(self.W1[0])
        enc_h = []
        for h in range(H):
            coeffs = [self.W1[f][h] for f in range(F)]
            tmp = self.eval_linear_wsum(enc_agg, coeffs)
            enc_h.append(tmp)
        print("Layer 1 liner finished")
 
        # square
        print("act started")
        for i in range(len(enc_h)):
            self.evaluator.square_inplace(enc_h[i])
            self.evaluator.relinearize_inplace(enc_h[i], self.relin_keys)
            self.evaluator.rescale_to_next_inplace(enc_h[i])
            op_counts["CMult"] += 1
        print("act finished")

        # aggregate
        print("Layer 2 agg starting")
        enc_agg = [self._aggregate(ct) for ct in enc_h]
        print("Layer 2 agg finished")

        # Layer2: linear
        print("Layer 2 linear starting")
        H = len(self.W2)
        C = len(self.W2[0])
        enc_out = []
        for c in range(C):
            coeffs = [self.W2[h][c] for h in range(H)]
            tmp = self.eval_linear_wsum(enc_agg, coeffs)
            enc_out.append(tmp)
        print("Layer 2 linear finished")

        return enc_out

# Decrypt outputs
def decrypt_outputs(enc_out, seal_ctx, N):
    dec = seal_ctx['decryptor']
    enc = seal_ctx['encoder']
    results = []
    for ct in enc_out:
        plain = dec.decrypt(ct)
        vec = enc.decode(plain)
        results.append(vec[:N])
    return results

# Main
if __name__ == '__main__':
    dataset, data = get_dataset(sys.argv[1])
    print(data.y.shape, len(data.y.shape))

    model = torch.load(f'./models/{sys.argv[1]}.pt')

    multilabel = False
    if len(data.y.shape) > 1:
        multilabel = True

    plain_acc = test_acc_calc(get_predictions(model(data.x, data.edge_index), multilabel), data)
    print(plain_acc)

    adj = get_A_bar(data.edge_index, data.num_nodes)
    seal_ctx = setup_seal(data.num_nodes)

    enc_gcn = EncGCN(seal_ctx, model)
    enc_gcn.encrypt_adj(adj.tolist())
    feats = data.x.t().tolist()
    enc_feats = enc_gcn.encrypt_features(feats)

    start = time.time()
    enc_out = enc_gcn.forward(enc_feats)
    elapsed = time.time() - start

    plain_out = torch.tensor(decrypt_outputs(enc_out, seal_ctx, data.num_nodes)).t()
    preds = get_predictions(plain_out, multilabel)
    test_accuracy = test_acc_calc(preds, data)

    print_op_counts()
    print(f"\nEncrypted latency = {elapsed:.2f}s, test acc = {test_accuracy:.4f}\n")

    with open("seal_utils/results.csv", 'a') as f:
        log = f"{sys.argv[1]},{test_accuracy},{plain_acc},{elapsed},{op_counts['Rot']},{op_counts['PMult']},{op_counts['CMult']},{op_counts['Add']}\n"
        f.write(log)
