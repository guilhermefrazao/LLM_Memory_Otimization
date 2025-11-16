import os
import json

def write_json(data, filename: str):
    os.makedirs("output", exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Resultado salvo em: {filename}")