import sys
import time
import math

import torch
from torch_geometric.datasets import Planetoid, KarateClub
from torch_geometric.transforms import RandomNodeSplit

from openfhe import Ciphertext
from openfhe import GenCryptoContext, CCParamsCKKSRNS, PKESchemeFeature, SecurityLevel, ScalingTechnique

from train import train_gcn, test
from utils import get_A_bar, get_predictions, test_acc_calc, get_dataset

# -----------------------------------------------------------------------------
# Operation Counters
# -----------------------------------------------------------------------------
op_counts = {"Rot": 0, "PMult": 0, "CMult": 0, "Add": 0}

# Utility to print counts
def print_op_counts():
    print("\n=== OpenFHE Homomorphic Op Counts ===")
    for k, v in op_counts.items():
        print(f"  {k:5s} : {v}")
    print("======================================\n")

def setup_ckks(num_nodes: int):
    # 1) CKKS params
    mult_depth = 7  # Adjust based on your computation needs
    security_level = SecurityLevel.HEStd_NotSet

    # 2. Define the encryption parameters
    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(mult_depth)
    params.SetSecurityLevel(security_level)
    params.SetRingDim(8192)
    params.SetFirstModSize(34)
    params.SetScalingModSize(30)

    # 2) CryptoContext
    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    # 3) KeyGen + MultKey
    key_pair = cc.KeyGen()
    cc.EvalMultKeyGen(key_pair.secretKey)

    rotation_keys = [-i for i in range(num_nodes)]
    cc.EvalRotateKeyGen(key_pair.secretKey, rotation_keys, key_pair.publicKey)

    return cc, key_pair.publicKey, key_pair.secretKey

class EncGCN:
    def __init__(self, cc, pk, torch_model):
        """
        cc:         CryptoContextCKKS
        pk:         PublicKey for encryption
        torch_model: your trained GCN
        """
        self.cc = cc
        self.pk = pk

        # pull out weight matrices as Python lists
        # conv.lin.weight is [out_feats × in_feats], so we transpose → [in_feats][out_feats]
        self.W1 = torch_model.conv1.lin.weight.T.detach().tolist()
        self.W2 = torch_model.conv2.lin.weight.T.detach().tolist()


    def make_adjacency_diagonals(self, A):
        self.N = len(A)
        diags = []
        for d in range(self.N):
            diag_d = [A[i][(i + d) % self.N] for i in range(self.N)]
            diags.append(diag_d)
        return diags

    def encrypt_matrix(self, matrix):
        return [
            self.cc.Encrypt(self.pk, self.cc.MakeCKKSPackedPlaintext(r))
            for r in matrix
        ]

    def _aggregate(self, ct_feat, enc_adj):
        self.adj_diags = enc_adj
        s = math.isqrt(self.N) + 1  # baby step size
        m = (self.N + s - 1) // s   # giant step count

        # Precompute baby steps (rotated inputs)
        baby_steps = {}
        for j in range(s):
            if j >= self.N:
                break
            rot = self.cc.EvalRotate(ct_feat, -j)
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
                tmp = self.cc.EvalMult(rotated, self.adj_diags[idx])
                op_counts["CMult"] += 1
                self.cc.RelinearizeInPlace(tmp)
                self.cc.RescaleInPlace(tmp)

                if inner_sum is None:
                    inner_sum = tmp
                else:
                    self.cc.EvalAddInPlace(inner_sum, tmp)
                    op_counts["Add"] += 1

            # Rotate inner sum by i*s
            if i > 0:
                inner_sum = self.cc.EvalRotate(inner_sum, -i * s)
                op_counts["Rot"] += 1

            if agg is None:
                agg = inner_sum
            else:
                self.cc.EvalAddInPlace(agg, inner_sum)
                op_counts["Add"] += 1

        return agg


    def _conv(self, enc_inputs, W):
        # enc_inputs: list of F ciphertexts
        # W:           Python list of shape [in_dim][out_dim]
        F = len(W)
        H = len(W[0])
        outs = []
        for h in range(H):
            coeffs = [W[f][h] for f in range(F)]
            ct_out = self.cc.EvalLinearWSum(enc_inputs, coeffs)
            outs.append(ct_out)
            op_counts["PMult"] += F
            op_counts["Add"] += F
        return outs


    def forward(self, enc_feats, enc_adj):
        # — Layer 1 — #
        # 1) Neighborhood aggregation: A · W1
        enc_agg = [self._aggregate(f, enc_adj) for f in enc_feats]
        print("First agg done")

        # 2) Linear map X' = X_agg · W1
        enc_h1 = self._conv(enc_agg, self.W1)
        print("First lin done")

        # 3) Polynomial activation (square)
        for i in range(len(enc_h1)):
            self.cc.EvalSquareInPlace(enc_h1[i])
            self.cc.RelinearizeInPlace(enc_h1[i])
            self.cc.RescaleInPlace(enc_h1[i])
            op_counts["CMult"] += 1
        print("act done")

        # — Layer 2 — #
        # 4) Aggregate again
        enc_agg_h1 = [self._aggregate(v, enc_adj) for v in enc_h1]
        print("2nd agg done")

        # 5) Linear map to outputs
        enc_out = self._conv(enc_agg_h1, self.W2)
        print("2nd lin done")

        return enc_out

def decrypt_outputs(enc_out, num_nodes, cc, secret_key) -> list[list[float]]:
    pt_out = []
    for ct in enc_out:
        decrypted_plaintext = cc.Decrypt(ct, secret_key)
        pt_out.append(decrypted_plaintext.GetRealPackedValue())
    return pt_out


# Main
if __name__ == '__main__':
    dataset, data = get_dataset(sys.argv[1])

    model = torch.load(f'./models/{sys.argv[1]}.pt')

    multilabel = False
    if len(data.y.shape) > 1:
        multilabel = True

    plain_acc = test_acc_calc(get_predictions(model(data.x, data.edge_index), multilabel), data)
    print(plain_acc)

    adj = get_A_bar(data.edge_index, data.num_nodes)
    cc, publicKey, privateKey = setup_ckks(data.num_nodes)

    enc_gcn = EncGCN(cc, publicKey, model)
    adj_diag = enc_gcn.make_adjacency_diagonals(adj.tolist())
    print("Encrypting Adj")
    enc_adj = enc_gcn.encrypt_matrix(adj_diag)
    print("finished Encrypting Adj")
    feats = data.x
    feats = feats.t()
    feats = feats.tolist()
    print("Encrypting x")
    enc_feats = enc_gcn.encrypt_matrix(feats)
    print("finished Encrypting x")

    start = time.time()
    enc_out = enc_gcn.forward(enc_feats, enc_adj)
    elapsed = time.time() - start

    plain_out = decrypt_outputs(enc_out, data.num_nodes, cc, privateKey)
    plain_out = torch.real(torch.tensor(plain_out)).t()[:data.num_nodes]


    preds = get_predictions(plain_out, multilabel)
    test_accuracy = test_acc_calc(preds, data)

    print_op_counts()
    print(f"\nEncrypted latency = {elapsed:.2f}s, test acc = {test_accuracy:.4f}\n")

    with open("openfhe_utils/results.csv", 'a') as f:
        log = f"{sys.argv[1]},{test_accuracy},{plain_acc},{elapsed},{op_counts['Rot']},{op_counts['PMult']},{op_counts['CMult']},{op_counts['Add']}\n"
        f.write(log)
