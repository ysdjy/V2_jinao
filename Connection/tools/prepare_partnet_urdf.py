#!/usr/bin/env python3
"""Prepare PartNet-Mobility URDF assets for Isaac Sim URDF import.

PartNet mesh filenames often contain hyphens (e.g. ``new-0.obj``), which produce
invalid USD prim paths during URDF import. This script creates underscore-safe
copies of meshes and a companion URDF that references them.
"""

from __future__ import annotations

import argparse
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def sanitize_name(name: str) -> str:
    """Replace characters that are invalid in USD prim paths."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def sanitize_filename(path: Path) -> str:
    """Sanitize the file stem while preserving the extension for mesh importers."""
    return sanitize_name(path.stem) + path.suffix.lower()


def prepare_asset(asset_dir: Path, urdf_name: str = "mobility.urdf", output_name: str = "mobility_isaac.urdf") -> Path:
    """Create sanitized mesh copies and an Isaac-ready URDF inside *asset_dir*."""
    asset_dir = asset_dir.resolve()
    urdf_path = asset_dir / urdf_name
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    mesh_dir = asset_dir / "textured_objs"
    isaac_mesh_dir = asset_dir / "textured_objs_isaac"
    if isaac_mesh_dir.exists():
        shutil.rmtree(isaac_mesh_dir)
    isaac_mesh_dir.mkdir()

    replacements: dict[str, str] = {}

    for src_path in sorted(mesh_dir.iterdir()):
        if src_path.suffix.lower() not in {".obj", ".mtl"}:
            continue
        dst_name = sanitize_filename(src_path)
        dst_path = isaac_mesh_dir / dst_name
        content = src_path.read_text(encoding="utf-8")
        if src_path.suffix.lower() == ".obj":
            content = re.sub(
                r"^mtllib\s+(.+)$",
                lambda match: f"mtllib {sanitize_filename(Path(match.group(1).strip()))}",
                content,
                flags=re.MULTILINE,
            )
        dst_path.write_text(content, encoding="utf-8")
        rel_src = f"textured_objs/{src_path.name}"
        rel_dst = f"textured_objs_isaac/{dst_name}"
        replacements[rel_src] = rel_dst

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    removed_meshes: list[str] = []
    for elem in root.iter():
        if "name" in elem.attrib:
            elem.attrib["name"] = sanitize_name(elem.attrib["name"])

    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for geom_elem in list(link.findall(tag)):
                mesh = geom_elem.find("./geometry/mesh")
                if mesh is None:
                    continue
                src = mesh.attrib.get("filename")
                if src is None:
                    continue
                if src in replacements:
                    mesh.attrib["filename"] = replacements[src]
                    continue
                if not (asset_dir / src).is_file():
                    link.remove(geom_elem)
                    removed_meshes.append(src)

    output_path = asset_dir / output_name
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    if removed_meshes:
        print("Removed URDF elements with missing meshes:")
        for mesh in sorted(set(removed_meshes)):
            print(f"  {mesh}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "asset_dir",
        type=Path,
        help="Path to a PartNet-Mobility asset directory (e.g. Connection/USD/12252).",
    )
    parser.add_argument("--urdf-name", default="mobility.urdf", help="Source URDF filename.")
    parser.add_argument(
        "--output-name",
        default="mobility_isaac.urdf",
        help="Sanitized URDF filename written into the asset directory.",
    )
    args = parser.parse_args()

    output_path = prepare_asset(args.asset_dir, args.urdf_name, args.output_name)
    print(f"Prepared Isaac URDF: {output_path}")


if __name__ == "__main__":
    main()
