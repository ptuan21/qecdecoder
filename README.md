# qecdecoder

A graph neural network decoder for quantum error correction, benchmarked
against the classical Minimum Weight Perfect Matching (MWPM) baseline.

This is an ongoing research project. The goal is to train a GNN decoder for
the surface code, rigorously compare it against MWPM (via
[PyMatching](https://github.com/oscarhiggott/PyMatching)) on logical error
rate and decode latency, and publish both the code and a benchmark
report/preprint.

## Status

**Phase 1 complete**: simulation + baseline pipeline validated. The
end-to-end `simulate -> decode -> benchmark` pipeline is tested against a
closed-form theoretical logical error rate for the repetition code (where
MWPM decoding is provably optimal and reduces to majority vote). This gives
confidence the pipeline is correct before moving to the surface code, where
no simple closed form exists.

**Next**: Phase 2, generating surface-code datasets under a code-capacity
noise model and running the full MWPM benchmark (logical error rate vs.
physical error rate, threshold estimate) as the baseline the GNN decoder
will be compared against.

See `.claude/plans/` (or ask) for the full roadmap.

## Stack

- [Stim](https://github.com/quantumlib/Stim) — circuit simulation & sampling
- [PyMatching](https://github.com/oscarhiggott/PyMatching) — MWPM baseline
- PyTorch + PyTorch Geometric — GNN decoder (CPU-first, GPU-ready)
- `uv` for dependency management

## Setup

```bash
uv sync
uv run pytest
```

## Layout

```
src/qecdecoder/
  codes.py       # Stim circuit builders (repetition code, rotated surface code)
  noise.py        # noise-model helpers (code-capacity depolarizing noise)
  simulate.py      # syndrome + logical-observable sampling
  baseline.py       # PyMatching MWPM wrapper
  benchmark.py       # empirical & theoretical logical error rate
tests/             # one test module per src module, plus integration tests
experiments/configs/  # per-experiment YAML configs (reproducibility)
experiments/results/  # experiment outputs (gitignored; kept locally)
paper/               # preprint/benchmark-report draft
```
