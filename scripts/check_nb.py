import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open("livertumor-model.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
print(f"Total cells: {len(cells)}")
for i, c in enumerate(cells):
    src = "".join(c.get("source", []))
    # Strip non-ASCII for display
    preview = "".join(ch if ord(ch) < 128 else '?' for ch in src[:90]).replace("\n", " ")
    print(f"  [{i:2d}] {c['cell_type']:8s} | {preview}")
