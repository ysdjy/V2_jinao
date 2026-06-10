#!/usr/bin/env python3
# Copyright (c) 2026. KitchenV0 project.
# SPDX-License-Identifier: BSD-3-Clause

"""Clean a PartNet-Mobility asset for Isaac Sim URDF import.

This is based on the older Connection/tools/clean_partnet_urdf.py workflow, with
one extra guard needed by microwave 7320: drop URDF visual/collision blocks whose
mesh files are missing from textured_objs/.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def sanitize(name: str) -> str:
    """Return a USD path friendly file or prim name."""

    return re.sub(r"[^A-Za-z0-9_.]", "_", name)


def clean_obj(text: str) -> str:
    """Sanitize material references in an OBJ file."""

    text = re.sub(r"mtllib\s+(\S+)", lambda m: "mtllib " + sanitize(m.group(1)), text)
    text = re.sub(r"usemtl\s+(\S+)", lambda m: "usemtl " + sanitize(m.group(1)), text)
    return text


def clean_mtl(text: str) -> str:
    """Sanitize material names in an MTL file."""

    return re.sub(r"newmtl\s+(\S+)", lambda m: "newmtl " + sanitize(m.group(1)), text)


def copy_clean_meshes(src_mesh_dir: Path, dst_mesh_dir: Path) -> set[str]:
    """Copy mesh files with sanitized names and return available relative paths."""

    dst_mesh_dir.mkdir(parents=True, exist_ok=True)
    available_meshes: set[str] = set()
    for src_path in sorted(src_mesh_dir.iterdir()):
        dst_name = sanitize(src_path.name)
        dst_path = dst_mesh_dir / dst_name
        if src_path.suffix.lower() == ".obj":
            dst_path.write_text(clean_obj(src_path.read_text(encoding="utf-8")), encoding="utf-8")
            available_meshes.add(f"textured_objs/{dst_name}")
        elif src_path.suffix.lower() == ".mtl":
            dst_path.write_text(clean_mtl(src_path.read_text(encoding="utf-8")), encoding="utf-8")
        else:
            shutil.copy2(src_path, dst_path)
    return available_meshes


def clean_urdf(src_urdf: Path, dst_urdf: Path, available_meshes: set[str]) -> None:
    """Sanitize URDF names and remove visual/collision blocks with missing meshes."""

    tree = ET.parse(src_urdf)
    robot = tree.getroot()

    for elem in robot.iter():
        name = elem.attrib.get("name")
        if name:
            elem.set("name", sanitize(name))

    for link in robot.findall("link"):
        for block_name in ("visual", "collision"):
            for block in list(link.findall(block_name)):
                mesh = block.find("./geometry/mesh")
                if mesh is None:
                    continue
                filename = mesh.attrib.get("filename")
                if not filename:
                    continue
                directory, basename = os.path.split(filename)
                cleaned_filename = os.path.join(directory, sanitize(basename))
                if cleaned_filename not in available_meshes:
                    link.remove(block)
                    continue
                mesh.set("filename", cleaned_filename)

    tree.write(dst_urdf, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="Source PartNet asset directory.")
    parser.add_argument("dst", type=Path, help="Destination cleaned asset directory.")
    parser.add_argument("--urdf", default="mobility.urdf", help="URDF filename inside the source directory.")
    args = parser.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)

    for entry in src.iterdir():
        if entry.name in {"textured_objs", args.urdf}:
            continue
        target = dst / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target)
        else:
            shutil.copy2(entry, target)

    available_meshes = copy_clean_meshes(src / "textured_objs", dst / "textured_objs")
    clean_urdf(src / args.urdf, dst / args.urdf, available_meshes)
    print(f"Cleaned PartNet asset: {dst}")


if __name__ == "__main__":
    main()
