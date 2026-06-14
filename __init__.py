"""Reduced differential KirchhoffNet implementation (idealized version).

Follows the reviewer's recommended architecture:
  - direct BPTT through fixed-step Heun integration
  - differential state x_j = v_j+ - v_j-
  - sparse COO graph stages with NE/Proj/cluster topologies
  - soft library selection over L/S/Z cell families
  - SimContext injects PVT + mismatch during training
  - input/output mappers are simple affine + tanh (no MLPs)

Every tunable constant lives in config.py.
"""
