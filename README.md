# Eidoku

Eidoku is a lightweight, post-hoc verification gate for LLM reasoning.
It detects structurally inconsistent reasoning steps that remain fluent and high-probability.
This repository provides a reproducible reference implementation corresponding to the accompanying paper.

---

## Overview

Eidoku evaluates whether a proposed reasoning step can be *structurally accommodated*
by a given context, without attempting semantic understanding or truth verification.

The system operates purely as a **verification gate**:
it does not generate reasoning, interpret meaning, or access model internals.

---

## Repository Contents

| File | Description |
|------|-------------|
| `eidoku.py` | Core implementation of the Eidoku verification gate |
| `generate_rgd.py` | Synthetic RGD dataset generator (noise-free) |
| `generate_rgd_with_noise.py` | RGD generator with injected contextual noise |
| `run_benchmark.py` | Reproduces main benchmark results (FTAR / TTAR) |
| `run_sensitivity_2d_with_no_noise.py` | Reproduces Appendix sensitivity analysis |

---

## Requirements

- Python 3.9+
- NumPy
- scikit-learn

No GPU is required.

---

## Quick Reproduction

### 1. Generate synthetic data

```bash
python generate_rgd_with_noise.py
```

This generates the noisy RGD dataset used in the main experiments.

### 2. Run benchmark evaluation
```bash
python run_benchmark.py
```

This reproduces the FTAR / TTAR results reported in the paper.

### 3. Sensitivity analysis (Appendix)
```bash
python run_sensitivity_2d_with_no_noise.py
```

This reproduces the 2D sensitivity plots for percentile and threshold parameters.

---

## What Eidoku Does

- Evaluates structural coherence of reasoning candidates

- Penalizes logical, structural, and contextual violations

- Operates independently of token probabilities or model internals

- Acts as a rejection gate, not a generator
---
## What Eidoku Does NOT Do (Non-Goals)

- Semantic understanding or truth verification

- Reasoning generation or chain-of-thought synthesis

- Use of attention weights, logits, or RLHF signals

- Philosophical or normative interpretation

##### Eidoku only determines whether a proposed reasoning step can be consistently embedded within the given context.
---
## Relation to the Paper

This codebase is a minimal, faithful implementation of the methods described in the paper.

All theoretical discussions beyond verification (e.g., broader semantic frameworks) are intentionally excluded from this repository and are not required to reproduce the results.

## Reproducibility Notes

- All experiments are deterministic given fixed random seeds.

- Parameter values used in the paper are explicitly reported.

- Sensitivity analyses are included to demonstrate robustness.

## License

Apache 2.0 License.
