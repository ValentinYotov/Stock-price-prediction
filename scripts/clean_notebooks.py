"""
Remove emojis and excessive prints from notebook source cells.
Run from project root: python scripts/clean_notebooks.py
"""
import json
import re
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F9FF"  # misc symbols and pictographs
    "\U00002702-\U000027B0"
    "\U0001F600-\U0001F64F"
    "\U0001F1E0-\U0001F1FF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "\u2300-\u23FF"
    "\u2B50"
    "\u231A"
    "\u231B"
    "\u25AA-\u25AB"
    "\u25B6"
    "\u25C0"
    "\u25FB-\u25FE"
    "\u2934-\u2935"
    "\u2194-\u2199"
    "\u21A9-\u21AA"
    "\u2B05-\u2B07"
    "\u2B1B-\u2B1C"
    "\u3030"
    "\u303D"
    "\u3297"
    "\u3299"
    "\U0001F004"
    "\U0001F0CF"
    "\u00A9"
    "\u00AE"
    "\u203C"
    "\u2049"
    "\u2122"
    "\u2139"
    "\u2194-\u2199"
    "\u21A9-\u21AA"
    "\u231A-\u231B"
    "\u23E9-\u23F3"
    "\u23F8-\u23FA"
    "\u24C2"
    "\u25AA-\u25AB"
    "\u25B6"
    "\u25C0"
    "\u25FB-\u25FE"
    "\u2614-\u2615"
    "\u2648-\u2653"
    "\u267F"
    "\u2693"
    "\u26A1"
    "\u26AA-\u26AB"
    "\u26BD-\u26BE"
    "\u26C4-\u26C5"
    "\u26CE"
    "\u26D4"
    "\u26EA"
    "\u26F2-\u26F3"
    "\u26F5"
    "\u26FA"
    "\u26FD"
    "\u2702"
    "\u2705"
    "\u2708-\u270D"
    "\u270F"
    "\u2712"
    "\u2714"
    "\u2716"
    "\u271D"
    "\u2721"
    "\u2728"
    "\u2733-\u2734"
    "\u2744"
    "\u2747"
    "\u274C"
    "\u274E"
    "\u2753-\u2755"
    "\u2757"
    "\u2763-\u2764"
    "\u2795-\u2797"
    "\u27A1"
    "\u27B0"
    "\u27BF"
    "\u2934-\u2935"
    "\u2B05-\u2B07"
    "\u2B1B-\u2B1C"
    "\u2B50"
    "\u2B55"
    "\u3030"
    "\u303D"
    "\u3297"
    "\u3299"
    "\U0001F000-\U0001F02F"
    "\U0001F0A0-\U0001F0FF"
    "\U0001F100-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F910-\U0001F92F"
    "\U0001F930-\U0001F9FF"
    "\uFE0F"  # variation selector (often after emoji)
    "]+",
    flags=re.UNICODE,
)


def clean_string(s: str) -> str:
    s = EMOJI_PATTERN.sub("", s)
    s = s.replace("ГРЕШКА", "Error")
    s = s.replace("Заредено!", "").replace("Филтрирано!", "")
    s = s.replace("Започване на обучение", "Training")
    s = s.replace("Обучението завърши успешно!", "Training completed.")
    s = s.replace("Обучението е прекъснато от потребителя.", "Training interrupted.")
    s = s.replace("КРИТИЧНА ГРЕШКА", "Error")
    s = s.replace("Грешка:", "Error:")
    s = s.replace("Грешка при", "Error")
    return s


def clean_notebook(path: Path, clear_outputs: bool = False) -> None:
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    changed = False
    for cell in nb.get("cells", []):
        if "source" in cell and isinstance(cell["source"], list):
            new_source = []
            for line in cell["source"]:
                if isinstance(line, str):
                    cleaned = clean_string(line)
                    if cleaned != line:
                        changed = True
                    new_source.append(cleaned)
                else:
                    new_source.append(line)
            cell["source"] = new_source
        if clear_outputs and "outputs" in cell and cell["outputs"]:
            cell["outputs"] = []
            changed = True
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"Cleaned: {path.name}")
    else:
        print(f"Skip (no changes): {path.name}")


def main():
    for path in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
        if path.name.startswith("."):
            continue
        try:
            clean_notebook(path)
        except Exception as e:
            print(f"Error {path.name}: {e}")


if __name__ == "__main__":
    main()
