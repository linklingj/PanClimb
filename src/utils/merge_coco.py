"""
Merge multiple COCO annotation files into one, fixing image/annotation ids.

Usage:
    python src/utils/merge_coco.py \\
        --inputs data/annotation/bh-coco.json data/annotation/bh-phone-coco.json data/annotation/sm-coco.json \\
        --output data/annotation/all-coco.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_categories_consistent(cats_list: List[List[Dict]]) -> List[Dict]:
    """
    Check that all category name->id mappings match.
    Returns the first list to use as canonical.
    """
    base = cats_list[0]
    base_map = {c["name"]: c["id"] for c in base}
    for cats in cats_list[1:]:
        other_map = {c["name"]: c["id"] for c in cats}
        if base_map != other_map:
            raise ValueError(f"Category mismatch: {base_map} vs {other_map}")
    return base


def merge_coco(files: List[Path]) -> Dict:
    data = [load_json(p) for p in files]
    categories = ensure_categories_consistent([d["categories"] for d in data])

    merged = {"images": [], "annotations": [], "categories": categories}
    next_image_id = 1
    next_ann_id = 1

    for src, d in zip(files, data):
        id_map: Dict[int, int] = {}

        for img in d.get("images", []):
            new_img = dict(img)
            new_img["id"] = next_image_id
            id_map[img["id"]] = next_image_id
            merged["images"].append(new_img)
            next_image_id += 1

        for ann in d.get("annotations", []):
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = id_map[ann["image_id"]]
            merged["annotations"].append(new_ann)
            next_ann_id += 1

        print(f"[OK] {src} -> +{len(d.get('images', []))} images, +{len(d.get('annotations', []))} anns")

    return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge COCO annotation files")
    p.add_argument("--inputs", "-i", nargs="+", required=True, help="Input COCO json files (order matters).")
    p.add_argument("--output", "-o", required=True, help="Output merged COCO json file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    inputs = [Path(p) for p in args.inputs]
    output = Path(args.output)

    merged = merge_coco(inputs)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Merged {len(inputs)} files -> {len(merged['images'])} images, {len(merged['annotations'])} annotations")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
