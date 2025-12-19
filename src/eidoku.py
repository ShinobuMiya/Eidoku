# ----------------------------
# Note: This script is a lightweight version that uses TF-IDF
# to minimize dependencies. To reproduce the paper,
# please use the full version using the transformers library.
# ----------------------------
from __future__ import annotations

import math
import re
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# ----------------------------
# Utilities: Robust Parser
# ----------------------------

# [FIX] Regex updated to handle spaces in entities (e.g., "A shark")
_IS_PATTERN = re.compile(
    r"^\s*(?P<subj>[\w\s\-\']+?)\s+is\s+(?P<pred>[\w\s\-\']+?)\s*\.?\s*$",
    re.IGNORECASE
)

# Logical connector detector
_THEREFORE_PATTERN = re.compile(
    r"^\s*(?:Therefore|Thus|Hence|So|Consequently),?\s+(?P<rest>.+?)\s*$",
    re.IGNORECASE
)

def parse_is_relation(sentence: str) -> Optional[Tuple[str, str]]:
    s = sentence.strip()
    s = _THEREFORE_PATTERN.sub(r"\g<rest>", s) # Remove "Therefore"
    m = _IS_PATTERN.match(s)
    if not m:
        return None
    return m.group("subj").strip().lower(), m.group("pred").strip().lower()

def build_knowledge_edges(sentences: List[str]) -> List[Tuple[str, str]]:
    edges = []
    for s in sentences:
        rel = parse_is_relation(s)
        if rel:
            edges.append(rel)
    return edges

def transitive_closure(edges: List[Tuple[str, str]]) -> Dict[str, set]:
    nodes = set()
    for a, b in edges:
        nodes.add(a)
        nodes.add(b)
    reach = {n: set() for n in nodes}
    for a, b in edges:
        reach[a].add(b)

    changed = True
    while changed:
        changed = False
        for a in list(reach.keys()):
            new = set(reach[a])
            for b in list(reach[a]):
                new |= reach.get(b, set())
            if not new.issubset(reach[a]):
                reach[a] |= new
                changed = True
    return reach

def get_all_nodes(edges: List[Tuple[str, str]]) -> set:
    nodes = set()
    for a, b in edges:
        nodes.add(a)
        nodes.add(b)
    return nodes

# ----------------------------
# Metric & Geometry
# ----------------------------

def compute_metric_tensor(context_vectors: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n, d = context_vectors.shape
    if n < 2:
        return np.eye(d, dtype=float)

    cov = np.cov(context_vectors, rowvar=False)
    cov = 0.5 * (cov + cov.T)
    w, V = np.linalg.eigh(cov)
    w = np.maximum(w, 0.0)
    stiffness = 1.0 / (w + eps)
    g = (V * stiffness) @ V.T
    g = 0.5 * (g + g.T)
    
    tr = np.trace(g)
    if tr > 0:
        g /= (tr / d + eps)
    return g

# ----------------------------
# Tension Proxies
# ----------------------------

def _tau_struct_mdl_proxy(
    step_a_text: str,
    step_b_text: str,
    known_edges: List[Tuple[str, str]],
) -> Tuple[float, str]:
    rel_b = parse_is_relation(step_b_text)
    if rel_b is None:
        return 8.0, "Syntax error / Unstructured"

    subj, pred = rel_b
    
    # 1. Boundary Check (Discontinuity)
    all_nodes = get_all_nodes(known_edges)
    # If the subject is completely new (not in context), it's a structural hallucination
    if subj not in all_nodes:
        return 10.0, f"Discontinuity: Subject '{subj}' not grounded in context"
    
    # 2. Path Check
    closure = transitive_closure(known_edges)
    if pred in closure.get(subj, set()):
        return math.log2(1.0 + 2.0), f"Path {subj}->...->{pred} exists"

    mids = closure.get(subj, set())
    found_2hop = any(pred in closure.get(mid, set()) for mid in mids)
    if found_2hop:
        return math.log2(1.0 + 4.0), f"2-hop bridge connecting {subj} to {pred}"

    return math.log2(1.0 + 10.0), f"No transitive path from {subj} to {pred}"

def _tau_curv_reconstruction_proxy(
    vec_b: np.ndarray,
    context_matrix: np.ndarray,
    metric_g: np.ndarray,
    k_max: int = 16,
    eps: float = 1e-12,
) -> Tuple[float, float, str]:
    d = vec_b.shape[0]
    if context_matrix.shape[0] < 2:
        val = float(vec_b.T @ metric_g @ vec_b)
        return math.log1p(val), val, "Insufficient context"

    X_centered = context_matrix - np.mean(context_matrix, axis=0)
    if np.sum(np.var(X_centered, axis=0)) < eps:
        val = float(vec_b.T @ metric_g @ vec_b)
        return math.log1p(val), val, "Low variance context"

    k = min(k_max, d, max(1, context_matrix.shape[0] - 1))
    pca = PCA(n_components=k, svd_solver="auto", random_state=0)
    pca.fit(context_matrix)
    
    proj = pca.inverse_transform(pca.transform(vec_b.reshape(1, -1))).reshape(-1)
    r = vec_b - proj
    
    raw_energy = float(r.T @ metric_g @ r)
    log_energy = math.log1p(raw_energy)
    
    reason = "Aligned"
    if raw_energy > 0.5: reason = "Minor deviation"
    if raw_energy > 1.0: reason = "Off-manifold outlier"
        
    return log_energy, raw_energy, reason

def _tau_nli_rule_proxy(
    step_b_text: str,
    known_edges: List[Tuple[str, str]],
) -> Tuple[float, str]:
    s = step_b_text.strip().lower()
    is_deduction = _THEREFORE_PATTERN.match(step_b_text) is not None

    if " not " in s or " isn't " in s:
        return 4.0, "Explicit negation"

    rel = parse_is_relation(step_b_text)
    if rel is None:
        return 2.0, "Unstructured"

    subj, pred = rel
    closure = transitive_closure(known_edges)

    if pred in closure.get(subj, set()):
        return 0.5, "Entailed"
    if subj in closure.get(pred, set()):
        return 3.0, "Cycle detected"
    
    if is_deduction:
        return 6.0, "Deductive fallacy: 'Therefore' used without entailment"
    
    return 2.0, "Not entailed"

# ----------------------------
# Eidoku Gate
# ----------------------------

@dataclass
class EidokuResult:
    candidate: List[str]
    accepted: bool
    rejected_at: Optional[int]
    reject_reason: Optional[str]
    J: float
    E_total: float
    tau_critical_used: float
    weights_used: Tuple[float, float, float]
    step_details: List[Dict[str, Any]]

class FastEidokuGate:
    def __init__(self, safety_margin: float = 1.1, auto_balance_weights: bool = True):
        self.safety_margin = float(safety_margin)
        self.auto_balance_weights = auto_balance_weights
        self.weights = (1.0, 1.0, 1.0)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    def _audit_inputs(self, candidate_steps: List[Any]) -> None:
        for step in candidate_steps:
            if not isinstance(step, str):
                raise TypeError(f"Input must be raw string. Received {type(step)}.")

    def _calculate_step_tension(
        self, step_A: str, step_B: str, vec_B: np.ndarray, context_matrix: np.ndarray,
        metric_g: np.ndarray, known_edges: List[Tuple[str, str]], weights: Tuple[float, float, float]
    ) -> Dict[str, Any]:
        
        w_s, w_c, w_l = weights
        val_struct, r_struct = _tau_struct_mdl_proxy(step_A, step_B, known_edges)
        val_curv, val_curv_raw, r_curv = _tau_curv_reconstruction_proxy(vec_B, context_matrix, metric_g)
        val_nli, r_nli = _tau_nli_rule_proxy(step_B, known_edges)

        tau_total = w_s * val_struct + w_c * val_curv + w_l * val_nli
        return {
            "tau": float(tau_total),
            "components": {
                "structure": {"value": float(val_struct), "reason": r_struct},
                "curvature": {"value": float(val_curv), "raw_value": float(val_curv_raw), "reason": r_curv},
                "logic":     {"value": float(val_nli), "reason": r_nli}
            }
        }

    def calibrate_weights_and_threshold(
        self, context_strs: List[str], context_vecs: np.ndarray, metric_g: np.ndarray
    ) -> Tuple[Tuple[float, float, float], float]:
        
        full_edges = build_knowledge_edges(context_strs)
        t_structs, t_curvs, t_logics = [], [], []
        
        for i in range(len(context_strs) - 1):
            s_a = context_strs[i]
            s_b = context_strs[i+1]
            vec_b = context_vecs[i+1]
            win_vecs = context_vecs[max(0, i-4):i+1]
            v_s, _ = _tau_struct_mdl_proxy(s_a, s_b, full_edges)
            v_c, _, _ = _tau_curv_reconstruction_proxy(vec_b, win_vecs, metric_g)
            v_n, _ = _tau_nli_rule_proxy(s_b, full_edges)
            t_structs.append(v_s)
            t_curvs.append(v_c)
            t_logics.append(v_n)
            
        if self.auto_balance_weights and t_structs:
            std_s = np.std(t_structs) + 1e-3
            std_c = np.std(t_curvs) + 1e-3
            std_l = np.std(t_logics) + 1e-3
            w_s = 1.0 / std_s
            w_c = 1.0 / std_c
            w_l = 1.0 / std_l
            avg = (w_s + w_c + w_l) / 3.0
            new_weights = (w_s/avg, w_c/avg, w_l/avg)
        else:
            new_weights = (1.0, 1.0, 1.0)
        
        # [Robustness Fix] If structure variance collapsed (perfect graph), force weight to 1.0
        # This handles synthetic data singularity.
        if new_weights[0] < 1e-3:
            lst = list(new_weights)
            lst[0] = 1.0
            new_weights = tuple(lst)

        total_tensions = []
        for i in range(len(t_structs)):
            val = new_weights[0]*t_structs[i] + new_weights[1]*t_curvs[i] + new_weights[2]*t_logics[i]
            total_tensions.append(val)
            
        if not total_tensions:
            tau_c = 4.0
        else:
            p95 = np.percentile(total_tensions, 95)
            tau_c = float(max(2.0, p95 * self.safety_margin))
            
        return new_weights, tau_c

    def analyze_sensitivity_values(self, context: List[str], candidates: List[List[str]]):
        all_texts = list(context)
        offsets = []
        curr = len(context)
        for cand in candidates:
            offsets.append((curr, len(cand)))
            all_texts.extend(cand)
            curr += len(cand)
            
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        all_vecs = vec.fit_transform(all_texts).toarray()
        
        ctx_vecs = all_vecs[:len(context)]
        g = compute_metric_tensor(ctx_vecs)
        
        calibrated_weights, _ = self.calibrate_weights_and_threshold(context, ctx_vecs, g)
        
        context_tensions = []
        current_edges = build_knowledge_edges(context)
        
        for i in range(1, len(context)):
            prev_text = context[i-1]
            step_text = context[i]
            step_vec = ctx_vecs[i]
            win_vecs = ctx_vecs[max(0, i-10):i]
            
            t_data = self._calculate_step_tension(
                prev_text, step_text, step_vec, win_vecs, g, current_edges, calibrated_weights
            )
            context_tensions.append(t_data["tau"])
            
            new_rel = parse_is_relation(step_text)
            if new_rel: current_edges.append(new_rel)

        candidates_max_tension = []
        
        for idx, cand_strs in enumerate(candidates):
            start, length = offsets[idx]
            cand_vecs = all_vecs[start : start+length]
            current_edges = build_knowledge_edges(context)
            history_vecs = ctx_vecs
            max_tau = 0.0
            
            for i, step_text in enumerate(cand_strs):
                step_vec = cand_vecs[i]
                prev_text = context[-1] if i == 0 else cand_strs[i-1]
                win_vecs = history_vecs[-10:]
                t_data = self._calculate_step_tension(
                    prev_text, step_text, step_vec, win_vecs, g, current_edges, calibrated_weights
                )
                if t_data["tau"] > max_tau:
                    max_tau = t_data["tau"]
                
                new_rel = parse_is_relation(step_text)
                if new_rel: current_edges.append(new_rel)
                history_vecs = np.vstack([history_vecs, step_vec])
            
            candidates_max_tension.append(max_tau)
            
        return context_tensions, candidates_max_tension

    def select_best(self, context: List[str], candidates: List[List[str]]) -> Tuple[EidokuResult, List[EidokuResult]]:
        all_texts = list(context)
        offsets = []
        curr = len(context)
        for cand in candidates:
            self._audit_inputs(cand)
            offsets.append((curr, len(cand)))
            all_texts.extend(cand)
            curr += len(cand)
            
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        all_vecs = vec.fit_transform(all_texts).toarray()
        ctx_vecs = all_vecs[:len(context)]
        g = compute_metric_tensor(ctx_vecs)
        calibrated_weights, tau_c = self.calibrate_weights_and_threshold(context, ctx_vecs, g)
        
        results = []
        for idx, cand_strs in enumerate(candidates):
            start, length = offsets[idx]
            cand_vecs = all_vecs[start : start+length]
            current_edges = build_knowledge_edges(context)
            history_vecs = ctx_vecs
            J = 0.0
            rejected_at = None
            reject_reason = None
            step_details = []
            
            for i, step_text in enumerate(cand_strs):
                step_vec = cand_vecs[i]
                prev_text = context[-1] if not i else cand_strs[i-1]
                win_vecs = history_vecs[-10:]
                t_data = self._calculate_step_tension(
                    prev_text, step_text, step_vec, win_vecs, g, current_edges, calibrated_weights
                )
                step_details.append(t_data)
                J += t_data["tau"]
                if t_data["tau"] > tau_c:
                    rejected_at = i
                    comps = t_data["components"]
                    vals = [comps[k]['value'] * calibrated_weights[n] for n, k in enumerate(["structure", "curvature", "logic"])]
                    max_key = list(comps.keys())[np.argmax(vals)]
                    reject_reason = (f"High Tension ({t_data['tau']:.2f} > {tau_c:.2f}) "
                                     f"triggered by {max_key}: {comps[max_key]['reason']}")
                    break
                new_rel = parse_is_relation(step_text)
                if new_rel: current_edges.append(new_rel)
                history_vecs = np.vstack([history_vecs, step_vec])
            
            E_total = J
            results.append(EidokuResult(
                candidate=cand_strs, accepted=(rejected_at is None), rejected_at=rejected_at,
                reject_reason=reject_reason, J=float(J), E_total=float(E_total),
                tau_critical_used=tau_c, weights_used=calibrated_weights, step_details=step_details
            ))
            
        results_sorted = sorted(results, key=lambda r: (not r.accepted, r.E_total))
        return results_sorted[0], results

# ----------------------------
# Demo (Self-Check)
# ----------------------------
if __name__ == "__main__":
    # Test the Shark/Fish case which was failing due to parser
    context = ["A shark is a fish.", "A fish is an animal."]
    c1 = ["Therefore A shark is an animal."]
    c2 = ["Therefore A shark is a movie."] # False target
    
    gate = FastEidokuGate(safety_margin=1.1, auto_balance_weights=True)
    print(f"--- Context: {context} ---")
    best, all_results = gate.select_best(context, [c1, c2])

    def print_hn_log(r: EidokuResult):
        print("\n" + "=" * 60)
        status = "[ACCEPTED]" if r.accepted else "[REJECTED]"
        print(f"Candidate: {r.candidate}  =>  {status}")
        print(f"Calibrated Threshold Used: τ_c = {r.tau_critical_used:.3f}")
        
        if not r.accepted:
            print(f"(!) REASON: {r.reject_reason}")
            
        # C term is now 0.000 (absorbed)
        print(f"Total Energy: {r.E_total:.3f} (J={r.J:.3f}, C=0.000)")
        
        print("\n[Detail: Step-wise Semantic Tension]")
        for i, step_d in enumerate(r.step_details):
            print(f"  Step {i}: τ = {step_d['tau']:.3f}")
            
            # Deep copy to modify for display without touching original
            comps_display = {k: v.copy() for k, v in step_d['components'].items()}
            
            # Format curvature to match requested style: "value (raw=...)"
            if "raw_value" in comps_display["curvature"]:
                c_dict = comps_display["curvature"]
                c_val = c_dict["value"]
                c_raw = c_dict["raw_value"]
                c_dict["value"] = f"{c_val:.3f} (raw={c_raw:.3f})"
                del c_dict["raw_value"] # remove key from display
            
            print(json.dumps(comps_display, indent=4))

    for r in all_results:
        print(f"\n[{'ACCEPTED' if r.accepted else 'REJECTED'}] {r.candidate}")
        if not r.accepted: print(f"  Reason: {r.reject_reason}")