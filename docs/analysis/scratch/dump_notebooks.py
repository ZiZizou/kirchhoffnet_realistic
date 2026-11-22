"""Dump notebook code cells and saved outputs to text files for reading."""
import json
import sys
from pathlib import Path

NBS = {
    "f2_smooth2d": Path(r"C:\Users\Atharva\Downloads\trained_knet\realizable-kirchhoffnet_f2_smooth2d.ipynb"),
    "deq_run": Path(r"C:\Users\Atharva\Downloads\trained_knet\realizable-kirchhoffnet_deq_run.ipynb"),
}
OUT = Path(r"C:\Users\Atharva\Documents\kirchhoffnet_realistic\tmp\analysis\scratch")

for tag, nb_path in NBS.items():
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    code_parts = []
    out_parts = []
    n_code = 0
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if c.get("cell_type") == "code":
            n_code += 1
            code_parts.append(f"\n# ===== CODE CELL {i} (code #{n_code}) =====\n{src}\n")
            for o in c.get("outputs", []):
                ot = o.get("output_type")
                if ot == "stream":
                    txt = "".join(o.get("text", []))
                    out_parts.append(f"\n# ===== CELL {i} STREAM =====\n{txt}\n")
                elif ot in ("execute_result", "display_data"):
                    data = o.get("data", {})
                    txt = "".join(data.get("text/plain", []))
                    out_parts.append(f"\n# ===== CELL {i} {ot} =====\n{txt}\n")
                elif ot == "error":
                    out_parts.append(f"\n# ===== CELL {i} ERROR =====\n"
                                     + "\n".join(o.get("traceback", [])) + "\n")
        else:
            code_parts.append(f"\n# ===== MARKDOWN CELL {i} =====\n{src}\n")
    (OUT / f"nb_{tag}_code.txt").write_text("".join(code_parts), encoding="utf-8")
    (OUT / f"nb_{tag}_outputs.txt").write_text("".join(out_parts), encoding="utf-8")
    md = nb.get("metadata", {})
    print(f"{tag}: {len(cells)} cells ({n_code} code), kernel={md.get('kernelspec', {}).get('name')}, "
          f"language={md.get('language_info', {}).get('version')}")
    print(f"  code.txt  {(OUT / f'nb_{tag}_code.txt').stat().st_size} bytes")
    print(f"  outputs.txt {(OUT / f'nb_{tag}_outputs.txt').stat().st_size} bytes")
