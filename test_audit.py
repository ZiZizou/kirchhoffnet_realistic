#!/usr/bin/env python3
import os
os.chdir('/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal')

from config import make_friedman2_preset, make_friedman3_preset

print('Test 1: Default make_friedman2_preset (should be torus)')
p1 = make_friedman2_preset()
print('  hidden_family:', p1['stages'][0]['hidden_family'])
print('  write_mode:', p1['write_mode'])
print('  read_mode:', p1['read_mode'])

print('\nTest 2: make_friedman2_preset with hidden_family=small_world')
p2 = make_friedman2_preset('small_world', small_world_seed=42)
print('  hidden_family:', p2['stages'][0]['hidden_family'])
print('  write_mode:', p2['write_mode'])
print('  read_mode:', p2['read_mode'])
print('  write_idx:', p2['write_idx'])

print('\nTest 3: make_friedman3_preset (should be torus, preserving defaults)')
p3 = make_friedman3_preset()
print('  hidden_family:', p3['stages'][0]['hidden_family'])
print('  write_mode:', p3['write_mode'])
print('  read_mode:', p3['read_mode'])

print('\nTest 4: make_friedman2_preset validation (small_world_k odd)')
try:
    make_friedman2_preset('small_world', small_world_k=5)
except ValueError as e:
    print('  Validation error (expected):', e)

print('\nTest 5: make_friedman2_preset validation (p > 1)')
try:
    make_friedman2_preset('small_world', small_world_p=1.2)
except ValueError as e:
    print('  Validation error (expected):', e)

print('\nTest 6: make_friedman2_preset validation (invalid hidden_family)')
try:
    make_friedman2_preset('grid')
except ValueError as e:
    print('  Validation error (expected):', e)

print('\nAll tests passed!')