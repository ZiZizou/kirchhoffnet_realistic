from config import make_friedman2_preset, make_friedman3_preset

# Test 1: Default (torus)
print('Test 1 - Default (torus):')
print('  hidden_family:', make_friedman2_preset()['hidden_family'])
print('  hidden_kwargs keys:', list(make_friedman2_preset()['stages'][0]['hidden_kwargs'].keys()))

# Test 2: Explicit small_world
print('\nTest 2 - Explicit small_world:')
p2 = make_friedman2_preset('small_world', small_world_k=4, small_world_p=0.3, small_world_seed=42)
print('  hidden_family:', p2['hidden_family'])
print('  hidden_kwargs:', p2['stages'][0]['hidden_kwargs'])

# Test 3: Small world with custom params
print('\nTest 3 - Custom small_world params:')
p3 = make_friedman2_preset('small_world', small_world_k=6, small_world_p=0.1, small_world_seed=99)
print('  hidden_kwargs:', p3['stages'][0]['hidden_kwargs'])

# Test 4: Validation error (odd k)
print('\nTest 4 - Validation error (odd k):')
try:
    make_friedman2_preset('small_world', small_world_k=5)
except ValueError as e:
    print('  Expected error:', e)

# Test 5: Validation error (p outside range)
print('\nTest 5 - Validation error (p > 1):')
try:
    make_friedman2_preset('small_world', small_world_p=1.5)
except ValueError as e:
    print('  Expected error:', e)

# Test 6: make_friedman3_preset() (should be torus)
print('\nTest 6 - make_friedman3_preset():')
print('  hidden_family:', make_friedman3_preset()['hidden_family'])

# Test 7: Invalid hidden_family
print('\nTest 7 - Invalid hidden_family:')
try:
    make_friedman2_preset('grid')
except ValueError as e:
    print('  Expected error:', e)