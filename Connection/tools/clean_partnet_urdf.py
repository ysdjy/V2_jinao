#!/usr/bin/env python3
# Copyright (c) 2026. Connection project.
# SPDX-License-Identifier: BSD-3-Clause

"""Sanitize a PartNet-Mobility asset so its URDF can be imported into USD.

PartNet meshes/materials use hyphens in their names (e.g. ``new-1.obj``,
``original-5.mtl``, visual ``name="door_frame-10"``). USD prim names cannot
contain hyphens, which makes the Isaac Sim URDF importer fail with
``Ill-formed SdfPath`` / ``Used null prim``.

This script copies the asset to a clean directory and replaces hyphens with
underscores in:
  * .obj / .mtl filenames
  * ``mtllib`` / ``usemtl`` references inside .obj files
  * ``newmtl`` definitions inside .mtl files
  * ``<mesh filename=...>`` and visual/collision ``name=`` attributes in the URDF

Usage:
    python clean_partnet_urdf.py <src_dir> <dst_dir> [--urdf mobility.urdf]
"""

import argparse
import os
import re
import shutil


def sanitize(name: str) -> str:
    return name.replace("-", "_")


def clean_obj(text: str) -> str:
    def fix_mtllib(m):
        base, ext = os.path.splitext(m.group(1))
        return "mtllib " + sanitize(base) + ext

    def fix_usemtl(m):
        return "usemtl " + sanitize(m.group(1))

    text = re.sub(r"mtllib\s+(\S+)", fix_mtllib, text)
    text = re.sub(r"usemtl\s+(\S+)", fix_usemtl, text)
    return text


def clean_mtl(text: str) -> str:
    def fix_newmtl(m):
        return "newmtl " + sanitize(m.group(1))

    return re.sub(r"newmtl\s+(\S+)", fix_newmtl, text)


def clean_urdf(text: str) -> str:
    # mesh filenames: keep directory, sanitize basename
    def fix_mesh(m):
        path = m.group(1)
        d, b = os.path.split(path)
        base, ext = os.path.splitext(b)
        new_path = os.path.join(d, sanitize(base) + ext)
        return f'filename="{new_path}"'

    text = re.sub(r'filename="([^"]+)"', fix_mesh, text)

    # visual/collision name attributes
    def fix_name(m):
        return f'name="{sanitize(m.group(1))}"'

    text = re.sub(r'name="([^"]+)"', fix_name, text)
    return text


def main():
    parser = argparse.ArgumentParser(description="Sanitize a PartNet-Mobility asset for URDF->USD import.")
    parser.add_argument("src", type=str, help="Source asset directory (contains the URDF and textured_objs/).")
    parser.add_argument("dst", type=str, help="Destination directory for the cleaned asset.")
    parser.add_argument("--urdf", type=str, default="mobility.urdf", help="URDF filename within src.")
    args = parser.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst, exist_ok=True)

    # copy everything except mesh dir and urdf (handled specially); keep images/ etc.
    for entry in os.listdir(src):
        s = os.path.join(src, entry)
        if entry == "textured_objs" or entry == args.urdf:
            continue
        d = os.path.join(dst, entry)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)

    # clean mesh directory
    mesh_src = os.path.join(src, "textured_objs")
    mesh_dst = os.path.join(dst, "textured_objs")
    os.makedirs(mesh_dst, exist_ok=True)
    n_obj = n_mtl = n_other = 0
    for f in os.listdir(mesh_src):
        base, ext = os.path.splitext(f)
        new_name = sanitize(base) + ext
        sp = os.path.join(mesh_src, f)
        dp = os.path.join(mesh_dst, new_name)
        if ext == ".obj":
            with open(sp, "r") as fp:
                content = fp.read()
            with open(dp, "w") as fp:
                fp.write(clean_obj(content))
            n_obj += 1
        elif ext == ".mtl":
            with open(sp, "r") as fp:
                content = fp.read()
            with open(dp, "w") as fp:
                fp.write(clean_mtl(content))
            n_mtl += 1
        else:
            shutil.copy2(sp, dp)
            n_other += 1

    # clean URDF
    with open(os.path.join(src, args.urdf), "r") as fp:
        urdf_text = fp.read()
    out_urdf = os.path.join(dst, args.urdf)
    with open(out_urdf, "w") as fp:
        fp.write(clean_urdf(urdf_text))

    print(f"[clean] cleaned {n_obj} obj, {n_mtl} mtl, {n_other} other mesh files")
    print(f"[clean] cleaned URDF -> {out_urdf}")


if __name__ == "__main__":
    main()
