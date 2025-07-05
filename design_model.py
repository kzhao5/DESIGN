import sys
import time
import math
import numpy as np
import torch

# Ensure gcn_model and utils are in the python path or same directory
from gcn_model import GCN
from utils import get_A_unnormalized, get_predictions, test_acc_calc, get_dataset

import seal
from seal import (
    EncryptionParameters,
    scheme_type,
    SEALContext,
    KeyGenerator,
    CKKSEncoder,
    Encryptor,
    Evaluator,
    Decryptor,
    Plaintext,
    Ciphertext,
    RelinKeys,
    GaloisKeys,
)

# Parameters for FHE ( potrebbero necessitare di aggiustamenti )
poly_modulus_degree = 8192 # Example, might need 16384 or more for depth
coeff_modulus = seal.CoeffModulus.Create(
    poly_modulus_degree, [40, 30, 30, 30, 30, 30, 40] # Adjusted for potentially more depth
)
scale = pow(2.0, 30)

# Operation Counters
op_counts = {"Rot": 0, "PMult": 0, "CMult": 0, "Add": 0, "PolyEval": 0, "MaskGen":0}

def reset_op_counts():
    global op_counts
    op_counts = {"Rot": 0, "PMult": 0, "CMult": 0, "Add": 0, "PolyEval": 0, "MaskGen":0}

def print_op_counts():
    print("\n=== DESIGN FHE Op Counts ===")
    for k, v in op_counts.items():
        print(f"  {k:5s} : {v}")
    print("======================================\n")


def setup_seal_context():
    parms = EncryptionParameters(scheme_type.ckks)
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(coeff_modulus)
    context = SEALContext(parms)

    keygen = KeyGenerator(context)
    public_key = keygen.create_public_key()
    secret_key = keygen.secret_key()
    relin_keys = keygen.create_relin_keys()
    # Need all rotations for general degree calculation and mask application
    galois_keys = keygen.create_galois_keys() # All powers of 2 by default

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


class EncGCN_DESIGN:
    def __init__(
        self,
        seal_params,
        torch_model,
        num_nodes,
        enable_pruning=True,
        enable_adaptive_activation=True,
        # Thresholds T_1 > T_2 > ... > T_m (for m levels of retained nodes)
        # Nodes with score < T_m are pruned by M0
        pruning_thresholds_T_plain=[2.0], # Example: 1 level for retained, T_m = 2.0
        # Polynomials for adaptive activation (coeffs for P_d1, P_d2, ..., P_dm)
        # Example: P_d1 = x^2, P_d2 = x
        poly_configs_D_coeffs=[[0,0,1]], # Coeffs for x^2
        # Polynomial for HE.AprxCmp(a,b) -> P(a-b) ~ 1 if a>=b, ~0 if a<b
        # Assumes (a-b) is scaled to [-1, 1]. P(x) = -0.25x^3 + 0.75x + 0.5
        cmp_poly_coeffs_plain=[0.5, 0.75, 0, -0.25], # c0, c1, c2, c3
    ):
        self.seal_ctx = seal_params
        self.encoder = seal_params['encoder']
        self.encryptor = seal_params['encryptor']
        self.evaluator = seal_params['evaluator']
        self.relin_keys = seal_params['relin_keys']
        self.galois_keys = seal_params['galois_keys']
        self.scale = scale
        self.N = num_nodes # Number of nodes

        self.W1 = torch_model.conv1.lin.weight.T.detach().tolist()
        self.W2 = torch_model.conv2.lin.weight.T.detach().tolist()

        self.enable_pruning = enable_pruning
        self.enable_adaptive_activation = enable_adaptive_activation
        self.pruning_thresholds_T_plain = sorted(pruning_thresholds_T_plain, reverse=True)
        self.poly_configs_D_coeffs = poly_configs_D_coeffs
        self.cmp_poly_coeffs_plain = cmp_poly_coeffs_plain

        if len(self.pruning_thresholds_T_plain) != len(self.poly_configs_D_coeffs) and enable_adaptive_activation:
            raise ValueError("Number of pruning thresholds must match number of polynomial configs for adaptive activation.")

        # Encrypt fixed values
        self.enc_one_vec = self.encryptor.encrypt(self.encoder.encode([1.0] * self.N, self.scale))
        self.enc_zero_vec = self.encryptor.encrypt(self.encoder.encode([0.0] * self.N, self.scale))

        self.enc_pruning_thresholds_T = []
        for t in self.pruning_thresholds_T_plain:
            pt = self.encoder.encode(t, self.scale) # Encode scalar threshold
            self.enc_pruning_thresholds_T.append(self.encryptor.encrypt(pt))


    def _eval_poly(self, enc_x, plain_coeffs):
        """Evaluates a polynomial P(x) = c0 + c1*x + c2*x^2 + ... homomorphically using Horner's method."""
        # P(x) = c0 + x(c1 + x(c2 + ... x(cn)))
        # Coeffs are [c0, c1, ..., cn]
        op_counts["PolyEval"] +=1
        if not plain_coeffs:
            return self.enc_zero_vec # Or handle error

        degree = len(plain_coeffs) - 1
        
        # Initialize with the highest degree coefficient cn
        # This needs to be a ciphertext of the same size as enc_x
        # For Horner's, result = coeffs[degree]
        # then result = enc_x * result + coeffs[degree-1]
        
        # Current result starts with the highest coefficient
        pt_coeff_d = self.encoder.encode(plain_coeffs[degree], self.scale)
        # We need to ensure it's a CT that can be multiplied by enc_x
        # A simple way is to encrypt it as a vector, or use multiply_plain if enc_x is at right level
        # For simplicity, let's assume we start with enc_result = enc_x * 0 + pt_coeff_d (if degree > 0)
        # or just pt_coeff_d if degree == 0
        
        if degree == 0:
            pt_c0 = self.encoder.encode(plain_coeffs[0], self.scale)
            enc_c0 = self.encryptor.encrypt(pt_c0) # encrypt as a vector
            # Ensure enc_c0 is at the same level as enc_x if we were to add them
            # For now, just return it, assuming it will be used correctly
            return enc_c0


        # Initialize with coeff[degree] * enc_x^(degree) or use Horner's
        # Horner: res = c_n
        # for i = n-1 down to 0: res = res * x + c_i
        
        # res starts as Enc(coeffs[degree])
        pt_cn = self.encoder.encode([plain_coeffs[degree]]*self.N, self.scale) # Encode as vector
        enc_res = self.encryptor.encrypt(pt_cn)
        self.evaluator.mod_switch_to_inplace(enc_res, enc_x.parms_id())


        for i in range(degree - 1, -1, -1):
            # res = res * x
            self.evaluator.mod_switch_to_inplace(enc_res, enc_x.parms_id()) # Ensure levels match for mult
            enc_res = self.evaluator.multiply(enc_res, enc_x)
            op_counts["CMult"] += 1
            self.evaluator.relinearize_inplace(enc_res, self.relin_keys)
            self.evaluator.rescale_to_next_inplace(enc_res)
            
            # res = res + c_i
            pt_ci = self.encoder.encode(plain_coeffs[i], self.scale) # c_i is scalar
            # Need to ensure pt_ci is at the correct level for add_plain
            # Or encrypt c_i as a vector and add ciphertexts
            current_res_parms_id = enc_res.parms_id()
            self.evaluator.mod_switch_to_inplace(pt_ci, current_res_parms_id)
            self.evaluator.add_plain_inplace(enc_res, pt_ci)
            op_counts["Add"] += 1
            # No rescale after add_plain usually, but noise grows.
            # SEAL handles noise in add_plain.

        return enc_res

    def _HE_AprxCmp(self, enc_a, enc_b_scalar_ct):
        """ Approximates (a >= b) -> Enc(1), (a < b) -> Enc(0)
            Computes P_cmp(enc_a - enc_b_scalar_ct)
            enc_b_scalar_ct is an encryption of a scalar threshold, needs to be subtracted from vector enc_a.
        """
        op_counts["MaskGen"] +=1
        # enc_diff = enc_a - enc_b_scalar_ct
        enc_diff = Ciphertext() # Create a new ciphertext for the result
        self.evaluator.sub(enc_a, enc_b_scalar_ct, enc_diff) # a - b
        op_counts["Add"] +=1 # Sub is an add

        # IMPORTANT: This assumes (enc_a - enc_b_scalar_ct) is scaled to approx [-1, 1]
        # for the specific cmp_poly_coeffs_plain = [-0.25x^3 + 0.75x + 0.5]
        # This is a strong assumption and likely needs input normalization or a different polynomial.
        return self._eval_poly(enc_diff, self.cmp_poly_coeffs_plain)


    def _generate_masks(self, enc_scores):
        """ Generates prune mask M0 and level masks M1, ..., Mm """
        num_levels = len(self.pruning_thresholds_T_plain) # This is 'm'
        enc_Tm = self.enc_pruning_thresholds_T[num_levels - 1]

        # M0_v = 1 if score_v < T_m  =>  M0 = 1 - HE.AprxCmp(scores, T_m)
        # HE.AprxCmp(scores, T_m) gives ~1 if scores >= T_m
        cmp_scores_gt_Tm = self._HE_AprxCmp(enc_scores, enc_Tm)
        enc_M0 = Ciphertext()
        self.evaluator.sub(self.enc_one_vec, cmp_scores_gt_Tm, enc_M0) # 1 - cmp
        op_counts["Add"] +=1
        # Ensure M0 is rescaled and at a good level if it's used in many multiplications
        # self.evaluator.rescale_to_next_inplace(enc_M0) # Optional, depends on P_cmp output scale

        enc_level_masks_M = []
        # M_i,v = 1 if T_i <= score_v < T_{i-1} (T_0 = infinity)
        # M_i = HE.AprxCmp(scores, T_i) * (1 - HE.AprxCmp(scores, T_{i-1}))

        # For M_1 (highest importance): score_v >= T_1
        # M_1 = HE.AprxCmp(scores, T_1)
        enc_T1 = self.enc_pruning_thresholds_T[0]
        enc_M1 = self._HE_AprxCmp(enc_scores, enc_T1)
        enc_level_masks_M.append(enc_M1)
        
        # Previous comparison result (for T_{i-1})
        prev_cmp_scores_gt_Ti_minus_1 = enc_M1 # This is HE.AprxCmp(scores, T1)

        for i in range(1, num_levels): # For M2 to Mm
            enc_Ti = self.enc_pruning_thresholds_T[i]
            
            # cmp_ge_Ti = HE.AprxCmp(scores, T_i)
            cmp_scores_gt_Ti = self._HE_AprxCmp(enc_scores, enc_Ti)
            
            # one_minus_cmp_lt_Ti_minus_1 = 1 - HE.AprxCmp(scores, T_{i-1})
            # This is actually (1 - prev_cmp_scores_gt_Ti_minus_1)
            not_prev_cmp = Ciphertext()
            self.evaluator.sub(self.enc_one_vec, prev_cmp_scores_gt_Ti_minus_1, not_prev_cmp)
            op_counts["Add"] +=1

            # M_i = cmp_ge_Ti * not_prev_cmp
            enc_Mi = self.evaluator.multiply(cmp_scores_gt_Ti, not_prev_cmp)
            op_counts["CMult"] +=1
            self.evaluator.relinearize_inplace(enc_Mi, self.relin_keys)
            self.evaluator.rescale_to_next_inplace(enc_Mi)
            enc_level_masks_M.append(enc_Mi)
            
            prev_cmp_scores_gt_Ti_minus_1 = cmp_scores_gt_Ti
            
        return enc_M0, enc_level_masks_M

    def encrypt_adj_diagonals(self, adj_matrix_plain): # adj_matrix_plain is a list of lists
        adj_diags_plain = []
        for d in range(self.N):
            diag = [adj_matrix_plain[i][(i + d) % self.N] for i in range(self.N)]
            adj_diags_plain.append(diag)
        
        self.enc_adj_diags = []
        for diag_plain in adj_diags_plain:
            pt = self.encoder.encode(diag_plain, self.scale)
            self.enc_adj_diags.append(self.encryptor.encrypt(pt))

    def encrypt_features(self, feats_plain_transposed): # feats_plain_transposed: list of F lists of length N
        enc_feats = []
        for fv in feats_plain_transposed:
            pt = self.encoder.encode(fv, self.scale)
            enc_feats.append(self.encryptor.encrypt(pt))
        return enc_feats

    def _aggregate(self, ct_feat_vector, current_adj_diags):
        # This is A_bar * x (or A_pruned * x)
        # ct_feat_vector is one feature vector (Ciphertext)
        # current_adj_diags is the list of encrypted diagonals of the matrix to use
        s = math.isqrt(self.N) + 1
        m = (self.N + s - 1) // s

        baby_steps = {}
        for j in range(s):
            if j >= self.N: break
            rot = self.evaluator.rotate_vector(ct_feat_vector, j, self.galois_keys)
            baby_steps[j] = rot
            op_counts["Rot"] += 1
        
        agg_result = Ciphertext()
        # Initialize agg_result to zero ciphertext at the correct level
        # A bit tricky, let's use the first computed term if possible, or add to Enc(0)
        first_term = True
        
        for i in range(m):
            inner_sum = Ciphertext()
            first_inner_term = True

            for j in range(s):
                idx = i * s + j
                if idx >= self.N: break
                
                rotated_feat = baby_steps[j] # This is x rotated by j
                adj_diag_val = current_adj_diags[idx] # This is a diagonal of A

                # Ensure levels match for multiplication
                self.evaluator.mod_switch_to_inplace(adj_diag_val, rotated_feat.parms_id())
                
                tmp = self.evaluator.multiply(rotated_feat, adj_diag_val)
                op_counts["CMult"] += 1
                self.evaluator.relinearize_inplace(tmp, self.relin_keys)
                self.evaluator.rescale_to_next_inplace(tmp)

                if first_inner_term:
                    inner_sum = tmp
                    first_inner_term = False
                else:
                    self.evaluator.mod_switch_to_inplace(tmp, inner_sum.parms_id())
                    self.evaluator.add_inplace(inner_sum, tmp)
                    op_counts["Add"] += 1
            
            if i > 0 and not first_inner_term: # if inner_sum was actually computed
                self.evaluator.rotate_vector_inplace(inner_sum, i * s, self.galois_keys)
                op_counts["Rot"] += 1

            if not first_inner_term: # if inner_sum has content
                if first_term:
                    agg_result = inner_sum
                    first_term = False
                else:
                    self.evaluator.mod_switch_to_inplace(inner_sum, agg_result.parms_id())
                    self.evaluator.add_inplace(agg_result, inner_sum)
                    op_counts["Add"] += 1
        
        if first_term: # If N=0 or something went wrong, return Enc(0)
             pt_zero = self.encoder.encode([0.0]*self.N, self.scale)
             return self.encryptor.encrypt(pt_zero)

        return agg_result

    def _eval_linear_wsum(self, enc_feat_list, weights_plain_matrix_col):
        # Computes sum_i weights[i] * enc_feat_list[i]
        # weights_plain_matrix_col is a list of floats (a column from W)
        out_ct = Ciphertext()
        # Initialize out_ct to Enc(0) at the right level
        # For simplicity, assign first product, then add.
        
        first_prod = True
        for i, enc_feat_i in enumerate(enc_feat_list):
            w_i = weights_plain_matrix_col[i]
            pt_w = self.encoder.encode(w_i, self.scale) # scalar weight
            
            self.evaluator.mod_switch_to_inplace(pt_w, enc_feat_i.parms_id())
            
            prod = self.evaluator.multiply_plain(enc_feat_i, pt_w)
            op_counts["PMult"] += 1
            # No relin after multiply_plain with fresh plaintext, but rescale is needed
            self.evaluator.rescale_to_next_inplace(prod) # Rescale to manage noise/scale

            if first_prod:
                out_ct = prod
                first_prod = False
            else:
                # Ensure levels match for addition
                self.evaluator.mod_switch_to_inplace(prod, out_ct.parms_id())
                self.evaluator.add_inplace(out_ct, prod)
                op_counts["Add"] += 1
        
        if first_prod: # if enc_feat_list was empty
            pt_zero = self.encoder.encode([0.0]*self.N, self.scale)
            return self.encryptor.encrypt(pt_zero)
        return out_ct

    def forward(self, initial_enc_feats, unnorm_adj_diags_for_degree, norm_adj_diags_for_conv):
        current_enc_feats = initial_enc_feats
        current_adj_diags_for_conv = norm_adj_diags_for_conv

        enc_M0 = None
        enc_level_masks_M = None

        if self.enable_pruning:
            print("Step 1: FHE Node Degree Computation (for pruning)")
            # Use unnormalized adjacency for degree calculation (A or A+I)
            # Assuming _aggregate(enc_one_vec, unnorm_adj_diags) gives row sums (degrees)
            enc_degrees = self._aggregate(self.enc_one_vec, unnorm_adj_diags_for_degree)
            print("...degrees computed.")

            print("Step 2: Mask Generation")
            enc_M0, enc_level_masks_M = self._generate_masks(enc_degrees)
            print("...masks generated.")

            print("Step 3: Apply Pruning")
            # Prune Features: X' = (1 - M0) * X
            keep_mask_X = Ciphertext()
            self.evaluator.sub(self.enc_one_vec, enc_M0, keep_mask_X) # keep_mask = 1 - M0
            op_counts["Add"]+=1
            
            pruned_enc_feats = []
            for feat_vec in current_enc_feats:
                self.evaluator.mod_switch_to_inplace(feat_vec, keep_mask_X.parms_id())
                pruned_vec = self.evaluator.multiply(feat_vec, keep_mask_X)
                op_counts["CMult"]+=1
                self.evaluator.relinearize_inplace(pruned_vec, self.relin_keys)
                self.evaluator.rescale_to_next_inplace(pruned_vec)
                pruned_enc_feats.append(pruned_vec)
            current_enc_feats = pruned_enc_feats
            print("...features pruned.")

            # Prune Adjacency Matrix: A'_uv = (1-M0_u)(1-M0_v)A_uv
            # keep_mask_A_u is keep_mask_X. keep_mask_A_v is rotated keep_mask_X.
            pruned_adj_diags = []
            for d, adj_diag_d in enumerate(current_adj_diags_for_conv):
                # term_A = adj_diag_d * keep_mask_X (for u index)
                self.evaluator.mod_switch_to_inplace(adj_diag_d, keep_mask_X.parms_id())
                term_A = self.evaluator.multiply(adj_diag_d, keep_mask_X)
                op_counts["CMult"]+=1
                self.evaluator.relinearize_inplace(term_A, self.relin_keys)
                self.evaluator.rescale_to_next_inplace(term_A)

                # keep_mask_v = rotate(keep_mask_X, d)
                keep_mask_v = self.evaluator.rotate_vector(keep_mask_X, d, self.galois_keys)
                op_counts["Rot"]+=1
                
                self.evaluator.mod_switch_to_inplace(term_A, keep_mask_v.parms_id())
                pruned_diag_d = self.evaluator.multiply(term_A, keep_mask_v)
                op_counts["CMult"]+=1
                self.evaluator.relinearize_inplace(pruned_diag_d, self.relin_keys)
                self.evaluator.rescale_to_next_inplace(pruned_diag_d)
                pruned_adj_diags.append(pruned_diag_d)
            current_adj_diags_for_conv = pruned_adj_diags
            print("...adjacency matrix pruned.")
        
        # --- Layer 1 ---
        print("Layer 1 GCN: Aggregation")
        enc_agg_l1 = [self._aggregate(feat_vec, current_adj_diags_for_conv) for feat_vec in current_enc_feats]
        
        print("Layer 1 GCN: Linear Transformation")
        F_in = len(self.W1)    # Input features to W1
        H_out = len(self.W1[0]) # Output features from W1
        enc_h_l1_pre_act = []
        for h_idx in range(H_out):
            w1_col_h = [self.W1[f_idx][h_idx] for f_idx in range(F_in)]
            enc_h_l1_pre_act.append(self._eval_linear_wsum(enc_agg_l1, w1_col_h))

        print("Layer 1 GCN: Activation")
        enc_h_l1_post_act = []
        if self.enable_adaptive_activation and self.enable_pruning and enc_level_masks_M is not None:
            # Apply adaptive activation: H = sum_i M_i * P_di(Z)
            # Initialize H_final = Enc(0)
            final_act_output_l1 = self.encryptor.encrypt(self.encoder.encode([0.0]*self.N, self.scale))

            for i, level_mask_Mi in enumerate(enc_level_masks_M):
                poly_coeffs_Pi = self.poly_configs_D_coeffs[i]
                for Z_vec in enc_h_l1_pre_act: # Apply to each feature component of Z
                    # P_di_Z = self._eval_poly(Z_vec, poly_coeffs_Pi) # This was wrong, apply poly to each Z component
                    # This logic needs to be per component of the hidden layer vector
                    # The loop for adaptive activation should be outside the loop for h_idx
                    pass # Corrected below

            # Corrected adaptive activation logic:
            temp_post_act_list = []
            for Z_vec_component in enc_h_l1_pre_act: # Z_vec_component is one of H_out feature vectors
                component_activated_sum = self.encryptor.encrypt(self.encoder.encode([0.0]*self.N, self.scale)) # Enc(0)
                first_level_act = True
                for i, level_mask_Mi in enumerate(enc_level_masks_M):
                    poly_coeffs_Pi = self.poly_configs_D_coeffs[i]
                    P_di_Z_component = self._eval_poly(Z_vec_component, poly_coeffs_Pi)

                    self.evaluator.mod_switch_to_inplace(P_di_Z_component, level_mask_Mi.parms_id())
                    term_Mi_P_di_Z = self.evaluator.multiply(P_di_Z_component, level_mask_Mi)
                    op_counts["CMult"]+=1
                    self.evaluator.relinearize_inplace(term_Mi_P_di_Z, self.relin_keys)
                    self.evaluator.rescale_to_next_inplace(term_Mi_P_di_Z)
                    
                    if first_level_act:
                        component_activated_sum = term_Mi_P_di_Z
                        first_level_act = False
                    else:
                        self.evaluator.mod_switch_to_inplace(term_Mi_P_di_Z, component_activated_sum.parms_id())
                        self.evaluator.add_inplace(component_activated_sum, term_Mi_P_di_Z)
                        op_counts["Add"]+=1
                temp_post_act_list.append(component_activated_sum)
            enc_h_l1_post_act = temp_post_act_list

        else: # Standard activation (x^2 as in original GCN model)
            for Z_vec_component in enc_h_l1_pre_act:
                act_res = self.evaluator.square(Z_vec_component)
                op_counts["CMult"] +=1 # Square is one CMult
                self.evaluator.relinearize_inplace(act_res, self.relin_keys)
                self.evaluator.rescale_to_next_inplace(act_res)
                enc_h_l1_post_act.append(act_res)
        
        # --- Layer 2 ---
        print("Layer 2 GCN: Aggregation")
        enc_agg_l2 = [self._aggregate(feat_vec, current_adj_diags_for_conv) for feat_vec in enc_h_l1_post_act]

        print("Layer 2 GCN: Linear Transformation (Output)")
        H_in_l2 = len(self.W2)
        C_out = len(self.W2[0])
        enc_outputs = []
        for c_idx in range(C_out):
            w2_col_c = [self.W2[h_idx][c_idx] for h_idx in range(H_in_l2)]
            enc_outputs.append(self._eval_linear_wsum(enc_agg_l2, w2_col_c))
            
        return enc_outputs

def decrypt_outputs(enc_out_list, seal_params, N):
    dec = seal_params['decryptor']
    enc = seal_params['encoder']
    results_plain = [] # List of plain vectors
    for ct in enc_out_list:
        plain = dec.decrypt(ct)
        vec = enc.decode(plain)
        results_plain.append(vec[:N]) # Get first N elements
    # Transpose to get N x C_out
    if not results_plain: return torch.tensor([])
    return torch.tensor(results_plain).t()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python design_model.py <dataset_name> [prune:on/off] [adaptive_act:on/off]")
        sys.exit(1)

    dataset_name = sys.argv[1]
    
    # Ablation study parameters from command line
    enable_pruning_arg = sys.argv[2].split(':')[1] == 'on' if len(sys.argv) > 2 else True
    enable_adaptive_act_arg = sys.argv[3].split(':')[1] == 'on' if len(sys.argv) > 3 else True

    print(f"Running DESIGN model for {dataset_name}")
    print(f"Pruning: {'Enabled' if enable_pruning_arg else 'Disabled'}")
    print(f"Adaptive Activation: {'Enabled' if enable_adaptive_act_arg else 'Disabled'}")

    reset_op_counts()
    dataset_obj, data = get_dataset(dataset_name)
    model = torch.load(f'./models/{dataset_name}.pt') # Load pretrained plaintext model

    multilabel = len(data.y.shape) > 1 and data.y.shape[1] > 1

    # --- DESIGN FHE Setup ---
    seal_ctx_params = setup_seal_context()
    
    # --- Get Adjacency Matrices ---
    # 1. Unnormalized Adjacency (A or A+I) for degree calculation
    adj_unnorm_plain_torch = get_A_unnormalized(data.edge_index, data.num_nodes, add_self_loops=True)
    adj_unnorm_plain_list = adj_unnorm_plain_torch.tolist()
    
    # 2. Normalized Adjacency (A_bar) for GCN convolutions
    # The _aggregate function expects diagonals of the matrix used in A*X
    # The original seal_model uses A_bar directly.
    # For DESIGN, this A_bar will be pruned if pruning is on.
    adj_norm_plain_torch = get_A_bar(data.edge_index, data.num_nodes)
    adj_norm_plain_list = adj_norm_plain_torch.tolist()

    # --- Hyperparameters for DESIGN ---
    # These need to be tuned based on dataset characteristics (e.g., degree distribution)
    # Example: For Cora, degrees might range up to 10-20.
    # If degrees are normalized (0-1), thresholds like [0.6, 0.2] might work.
    # For now, using example absolute degree thresholds.
    # For adaptive activation, if 2 levels of retained nodes:
    # Level 1 (score >= T1): Use poly_config[0]
    # Level 2 (T2 <= score < T1): Use poly_config[1]
    # Pruned (score < T2)
    
    # Example config: 2 levels of importance for retained nodes
    # Thresholds: T1=5 (high importance), T2=2 (medium importance, also prune threshold)
    pruning_thresholds = [5.0, 2.0]
    # Poly configs: x^2 for high, x for medium
    adaptive_poly_coeffs = [
        [0, 0, 1.0],  # x^2 for level 1 (score >= 5.0)
        [0, 1.0],     # x   for level 2 (2.0 <= score < 5.0)
    ]
    # If only one threshold, e.g., [2.0], then one level for retained (score >= 2.0)
    # and one poly_config.
    if dataset_name == "Karate": # Karate club has very low degrees
        pruning_thresholds = [3.0, 1.0]


    enc_gcn_design = EncGCN_DESIGN(
        seal_ctx_params,
        model,
        data.num_nodes,
        enable_pruning=enable_pruning_arg,
        enable_adaptive_activation=enable_adaptive_act_arg,
        pruning_thresholds_T_plain=pruning_thresholds,
        poly_configs_D_coeffs=adaptive_poly_coeffs
    )

    # Encrypt Adjacency Diagonals
    # For degree calculation (unnormalized)
    enc_gcn_design.encrypt_adj_diagonals(adj_unnorm_plain_list) # Stores in self.enc_adj_diags
    unnorm_adj_diags_for_degree = enc_gcn_design.enc_adj_diags # Keep a reference

    # For GCN convolutions (normalized)
    enc_gcn_design.encrypt_adj_diagonals(adj_norm_plain_list) # Overwrites self.enc_adj_diags
    norm_adj_diags_for_conv = enc_gcn_design.enc_adj_diags

    # Encrypt Features
    feats_plain_transposed = data.x.t().tolist()
    initial_enc_features = enc_gcn_design.encrypt_features(feats_plain_transposed)

    # --- FHE Inference ---
    print("Starting DESIGN FHE GCN inference...")
    start_time = time.time()
    enc_output_vectors = enc_gcn_design.forward(
        initial_enc_features,
        unnorm_adj_diags_for_degree,
        norm_adj_diags_for_conv
    )
    elapsed_time = time.time() - start_time
    print(f"DESIGN FHE GCN inference finished in {elapsed_time:.2f}s")

    # Decrypt and Evaluate
    decrypted_output_tensor = decrypt_outputs(enc_output_vectors, seal_ctx_params, data.num_nodes)
    
    # Calculate accuracy on test set
    # Ensure data has test_mask, or define how to evaluate
    if hasattr(data, 'test_mask'):
        preds_fhe = get_predictions(decrypted_output_tensor, multilabel)
        test_accuracy_fhe = test_acc_calc(preds_fhe, data) # test_acc_calc uses data.test_mask
    else: # Fallback if no test_mask, evaluate on all nodes (less ideal)
        print("Warning: No test_mask found in data. Evaluating on all nodes.")
        data.test_mask = torch.ones(data.num_nodes, dtype=torch.bool) # temp mask
        preds_fhe = get_predictions(decrypted_output_tensor, multilabel)
        test_accuracy_fhe = test_acc_calc(preds_fhe, data)


    # Plaintext model accuracy for reference
    model.eval()
    with torch.no_grad():
        plain_out_ref = model(data.x, data.edge_index)
    preds_plain_ref = get_predictions(plain_out_ref, multilabel)
    test_accuracy_plain_ref = test_acc_calc(preds_plain_ref, data)


    print_op_counts()
    print(f"\nDataset: {dataset_name}")
    print(f"Plaintext Model Test Accuracy: {test_accuracy_plain_ref:.4f}")
    print(f"DESIGN FHE Model Test Accuracy: {test_accuracy_fhe:.4f}")
    print(f"DESIGN FHE Inference Latency: {elapsed_time:.2f}s")

    # Log results
    with open("design_results.csv", 'a') as f:
        if f.tell() == 0: # Write header if file is new/empty
            f.write("Dataset,Pruning,AdaptiveAct,PlainAcc,FHEAcc,Latency,Rot,PMult,CMult,Add,PolyEval,MaskGen\n")
        log_entry = (
            f"{dataset_name},{enable_pruning_arg},{enable_adaptive_act_arg},"
            f"{test_accuracy_plain_ref:.4f},{test_accuracy_fhe:.4f},{elapsed_time:.2f},"
            f"{op_counts['Rot']},{op_counts['PMult']},{op_counts['CMult']},{op_counts['Add']},"
            f"{op_counts['PolyEval']},{op_counts['MaskGen']}\n"
        )
        f.write(log_entry)

