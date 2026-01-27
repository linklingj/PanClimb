"""
Convert VIA annotations (JSON or CSV) to COCO instance segmentation format.

Example:
    python src/via_to_coco.py \\
        --ann data/annotation/bh-annotation.json \\
        --images data/images/bh \\
        --output data/annotation/bh-coco.json
"""

import argparse
import csv
import itertools
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from PIL import Image


def load_via_annotations(path: Path) -> List[Dict[str, Any]]:
    """Load VIA annotations from JSON or CSV into a normalized list."""
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_via_json(path)
    if suffix == ".csv":
        return _load_via_csv(path)
    raise ValueError(f"Unsupported annotation format: {suffix}")


def _load_via_json(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "_via_img_metadata" in data:
        img_meta = data["_via_img_metadata"]
        if isinstance(img_meta, dict):
            order = data.get("_via_image_id_list") or list(img_meta.keys())
            return [img_meta[k] for k in order if k in img_meta]
        if isinstance(img_meta, list):
            return img_meta

    if isinstance(data, list):
        return data

    raise ValueError("Unrecognized VIA JSON structure.")


def _load_via_csv(path: Path) -> List[Dict[str, Any]]:
    by_file: Dict[str, Dict[str, Any]] = {}

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            entry = by_file.setdefault(
                filename,
                {
                    "filename": filename,
                    "size": int(row.get("file_size") or 0),
                    "regions": [],
                    "file_attributes": _safe_json(row.get("file_attributes") or "{}"),
                },
            )
            shape = _safe_json(row.get("region_shape_attributes") or "{}")
            attrs = _safe_json(row.get("region_attributes") or "{}")
            entry["regions"].append(
                {"shape_attributes": shape, "region_attributes": attrs}
            )

    return list(by_file.values())


def _safe_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logging.warning("Could not decode JSON fragment: %s", raw)
        return {}


def polygon_area(xs: Iterable[float], ys: Iterable[float]) -> float:
    x_list = list(xs)
    y_list = list(ys)
    if len(x_list) != len(y_list):
        raise ValueError("Polygon coordinate lengths do not match.")
    n = len(x_list)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += x_list[i] * y_list[j] - x_list[j] * y_list[i]
    return abs(area) / 2.0


def bbox_from_points(xs: Iterable[float], ys: Iterable[float]) -> List[float]:
    x_list = list(xs)
    y_list = list(ys)
    return [
        float(min(x_list)),
        float(min(y_list)),
        float(max(x_list) - min(x_list)),
        float(max(y_list) - min(y_list)),
    ]


def region_to_coco(region: Dict[str, Any]) -> Tuple[List[List[float]], List[float], float]:
    shape = region.get("shape_attributes", {})
    name = shape.get("name")

    if name == "polygon":
        xs = shape.get("all_points_x") or []
        ys = shape.get("all_points_y") or []
        if len(xs) < 3 or len(ys) < 3:
            raise ValueError("Polygon needs at least 3 points.")
        segmentation = [
            list(
                itertools.chain.from_iterable(
                    zip([float(x) for x in xs], [float(y) for y in ys])
                )
            )
        ]
        bbox = bbox_from_points(xs, ys)
        area = polygon_area(xs, ys)
        return segmentation, bbox, area

    if name == "rect":
        x = float(shape.get("x", 0))
        y = float(shape.get("y", 0))
        w = float(shape.get("width", 0))
        h = float(shape.get("height", 0))
        segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
        bbox = [x, y, w, h]
        area = w * h
        return segmentation, bbox, area

    raise ValueError(f"Unsupported shape type: {name}")


def build_coco(
    via_items: List[Dict[str, Any]],
    image_dir: Path,
    category_key: str = "hold_type",
    default_category: str = "unknown",
    supercategory: str = "climbing",
) -> Dict[str, Any]:
    coco_images: List[Dict[str, Any]] = []
    coco_annotations: List[Dict[str, Any]] = []
    category_to_id: Dict[str, int] = {}

    ann_id = 1
    img_id = 1

    for item in via_items:
        filename = item.get("filename")
        if not filename:
            logging.warning("Skipping entry without filename: %s", item)
            continue

        img_path = image_dir / filename
        if not img_path.exists():
            logging.warning("Image not found for entry: %s", img_path)
            continue

        with Image.open(img_path) as img:
            width, height = img.size

        coco_images.append(
            {
                "id": img_id,
                "file_name": filename,
                "width": width,
                "height": height,
            }
        )

        for region in item.get("regions", []):
            try:
                segmentation, bbox, area = region_to_coco(region)
            except ValueError as exc:
                logging.warning("Skipping region in %s: %s", filename, exc)
                continue

            region_attrs = region.get("region_attributes", {}) or {}
            category_name = region_attrs.get(category_key, default_category) or default_category
            if category_name not in category_to_id:
                category_to_id[category_name] = len(category_to_id) + 1

            extra_attrs = {k: v for k, v in region_attrs.items() if k != category_key}

            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_to_id[category_name],
                    "segmentation": segmentation,
                    "bbox": bbox,
                    "area": float(area),
                    "iscrowd": 0,
                    **({"attributes": extra_attrs} if extra_attrs else {}),
                }
            )
            ann_id += 1

        img_id += 1

    coco_categories = [
        {"id": cid, "name": name, "supercategory": supercategory}
        for name, cid in sorted(category_to_id.items(), key=lambda kv: kv[1])
    ]

    return {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VIA annotations to COCO.")
    parser.add_argument("--ann", required=True, help="Path to VIA JSON or CSV file.")
    parser.add_argument("--images", required=True, help="Directory containing images.")
    parser.add_argument("--output", required=True, help="Output COCO JSON path.")
    parser.add_argument(
        "--category-key",
        default="hold_type",
        help="Region attribute key used as category name (default: hold_type).",
    )
    parser.add_argument(
        "--default-category",
        default="unknown",
        help="Fallback category name when the key is missing.",
    )
    parser.add_argument(
        "--supercategory",
        default="climbing",
        help="Supercategory value for COCO categories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ann_path = Path(args.ann)
    image_dir = Path(args.images)
    output_path = Path(args.output)

    logging.info("Loading VIA annotations from %s", ann_path)
    via_items = load_via_annotations(ann_path)
    logging.info("Found %d image entries", len(via_items))

    logging.info("Building COCO structure...")
    coco = build_coco(
        via_items,
        image_dir=image_dir,
        category_key=args.category_key,
        default_category=args.default_category,
        supercategory=args.supercategory,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    logging.info(
        "Wrote COCO file to %s (images=%d, annotations=%d, categories=%d)",
        output_path,
        len(coco["images"]),
        len(coco["annotations"]),
        len(coco["categories"]),
    )


if __name__ == "__main__":
    main()
