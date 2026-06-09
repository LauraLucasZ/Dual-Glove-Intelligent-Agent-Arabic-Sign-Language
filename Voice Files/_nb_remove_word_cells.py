import json
from pathlib import Path

nb_path = Path("full_voice_prediction_pipeline.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

to_remove = []
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if src.startswith("## Build Words from Predicted Letters"):
        to_remove.append(i)
    elif src.startswith("WORD_GESTURES"):
        to_remove.append(i)

for i in sorted(to_remove, reverse=True):
    del nb["cells"][i]
    print("Removed cell", i)

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
