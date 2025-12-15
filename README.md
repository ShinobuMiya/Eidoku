# Eidoku (影読): Structural Verification for System 2 Reasoning

**A lightweight, structural verification layer for detecting "smooth falsehoods" in LLM reasoning chains.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-RFC-orange.svg)]()

> **"Reasoning that preserves context is cheap; reasoning that breaks context is expensive."**

Eidoku is a **System 2 verification framework** that evaluates candidate reasoning paths by measuring **semantic tension**—the structural cost of maintaining contextual continuity while accommodating a hypothesis.  
It acts as a **geometric gate** that rejects high-probability hallucinations that probabilistic methods (such as Best-of-N) often fail to detect.

[**📄 Read the Full Design Specification (PDF)**](spec/System2_Architecture.pdf)

---

## 🚨 The Problem: Why Best-of-N Isn't Enough

Current reasoning mitigations (Best-of-N, Self-Consistency, majority voting) rely on the assumption that errors behave like random noise.  
They systematically fail when models collapse into **confident but smooth falsehoods**.

If a model generates the same plausible error multiple times, Best-of-N merely reinforces it as consensus.

- **System 1 Failure:** Smooth falsehoods often have high likelihood because they are grammatically fluent and locally coherent.
- **The Gap:** We lack a metric **orthogonal to probability** that detects when a reasoning chain is structurally broken despite being likely.

---

## 🛠️ The Solution: Eidoku

Eidoku introduces a **structural tension metric (τ)**.

It does **not** attempt to determine objective truth.  
Instead, it enforces a simple structural constraint:

> **Breaking context must cost more than preserving it.**

Reasoning paths that violate this constraint accumulate tension and are rejected, even if they are statistically likely.

---

## 🧱 Architecture

Eidoku operates as a **post-hoc or stepwise verification layer**, requiring **no retraining** of the base model.

1. **System 1 (Generator):**  
   Produces candidate reasoning paths using standard Chain-of-Thought or sampling.

2. **System 2 (Eidoku):**  
   Scores each candidate using accumulated semantic tension (τ) and an explicit context-breaking penalty (C).

3. **Selection:**  
   Candidates are ranked by minimizing total structural cost.

```math
S^* = \arg \min_{S} (J + \lambda C)
```
where
J is accumulated semantic tension
C penalizes explicit context breaks or shortcuts


---

## 🔍 Comparison: Probabilistic vs. Structural Verification
|Feature|Best-of-N / Self-Consistency|Eidoku (Structural Verification)|
|:-----------|:------------|:------------|
|Primary Objective|Maximize likelihood / consensus|Minimize semantic tension under preserved context|
|Failure Mode|Smooth falsehoods survive reranking|Smooth falsehoods incur high structural cost|
|Hallucination Signal|Implicit (confidence calibration)|Explicit (tension spikes)|
|Compute Cost|Grows linearly with N|Applied selectively at high-risk branching points (gatekeeper mode)|
|Training Required|Sometimes|None|

---

## ⚙️ How It Works (Minimal Implementation)
You do not need to implement the full geometric language (Catelingo) to benefit from Eidoku.
A minimal prototype can be built using existing NLP tools to approximate structural cost via graph complexity or Minimum Description Length (MDL).
Note: Full Catelingo integration improves recall against temporally or causally constrained falsehoods,
but it is not required for initial validation or experimentation.

---

## Core Heuristic
A hallucinated conceptual bridge typically requires a more complex, ad-hoc structure to justify than a valid deductive step.

```python

# Pseudo-code for structural tension scoring
def calculate_structural_tension(statement_prev, statement_next):
    # 1. Extract semantic graphs
    G_prev = extract_graph(statement_prev)
    G_next = extract_graph(statement_next)
    
    # 2. Compute minimal connector that preserves constraints
    G_bridge = minimal_connector(G_prev, G_next)
    
    # 3. Tension grows with structural complexity
    tension = log2(1 + complexity(G_bridge))
    
    return tension

def verify_chain(chain):
    total_tension = 0
    TAU_CRITICAL = 2.0  # default threshold for logical tasks

    for step in chain:
        tau = calculate_structural_tension(step.prev, step.curr)
        total_tension += tau
        
        # Abort early if a spike occurs
        if tau > TAU_CRITICAL:
            return Reject(step)
            
    return Accept(chain)
```

---

## ⚠️ What This Is NOT

- Not a Truth Oracle

  Eidoku does not verify facts against the external world unless explicitly connected to RAG.
It checks structural consistency, not ground truth.

- Not a Consciousness Model

  While inspired by the Critical Projection Theory (CPM), Eidoku is an engineering application.
It simulates System 2–like verification by enforcing virtual structural constraints—nothing more.

---

### 🗺️ Roadmap & RFC

This project is proposed as an RFC (Request for Comments).

Suggested evaluation targets:

- OpenAI: Apply Eidoku to o1-style reasoning traces.
- Anthropic: Integrate as a constitutional constraint layer.
- Google / DeepMind: Validate on AlphaProof-style mathematical reasoning.
- Open Source: Reference implementation using standard Transformer tooling.

---

## 📜 License

This project is licensed under the Apache 2.0 License.

Based on the theoretical proposal
“Catelingo and Eidoku: A Practical Framework for Simulated System 2 Reasoning via Structural Tension Minimization”
(Miya, 2025)
