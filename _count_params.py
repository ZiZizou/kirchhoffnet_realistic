from pathlib import Path
import torch

root = Path(r'C:\Users\Atharva\Downloads\results_many_models_realizable_knet_nb')

print("=== KNet model.pt state_dict keys ===")
for d in sorted(root.iterdir()):
    if not d.is_dir() or '_knet' not in d.name:
        continue
    mp = d / 'model.pt'
    if mp.is_file():
        sd = torch.load(mp, map_location='cpu', weights_only=True)
        n_total = sum(int(p.numel()) for p in sd.values())
        print(f'\n{d.name:30s} total_state_dict={n_total:,}')
        for k, v in sorted(sd.items()):
            shape_str = str(list(v.shape))
            print(f'    {k:60s} {shape_str:25s} {v.numel():>6,}')
