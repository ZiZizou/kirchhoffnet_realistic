"""Reduced differential KirchhoffNet implementation.

Follows the reviewer's recommended architecture:
  - direct BPTT through fixed-step Heun integration
  - differential state x_j = v_j+ - v_j-
  - sparse COO graph stages with NE/Proj topologies (line, ring, grid, small_world, torus, empty)
  - per-edge analog devices (tanh, relu, realistic, free-tanh families)
  - SimContext injects PVT + mismatch during training
  - input/output mappers are simple affine + tanh (no MLPs)

Every tunable constant lives in config.py.
"""
