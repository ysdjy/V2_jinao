#!/usr/bin/env python3
"""Convert SAPIEN/PartNet-Mobility assets to Isaac Lab USD assets.

Expected layout, relative to this file:

    SapienAssetPipeline/
      raw_sapien/<asset_id>/        # downloaded from https://sapien.ucsd.edu/browse
      usd_assets/<Name_asset_id>/   # generated Isaac/Omniverse USD assets

Typical usage:

    python SapienAssetPipeline/tools/convert_sapien_asset.py convert 44853 --name Cabinet --usd-name cabinet.usd

For a newly downloaded asset:

    unzip 12345.zip -d SapienAssetPipeline/raw_sapien/12345
    python SapienAssetPipeline/tools/convert_sapien_asset.py convert 12345 --name MyObject
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path


DEFAULT_URDF_NAME = "mobility.urdf"
DEFAULT_PREPARED_URDF_NAME = "mobility_isaac.urdf"

KNOWN_ASSET_NAMES = {
    "101054": ("Knife", "knife.usd"),
    "12252": ("Fridge", "fridge.usd"),
    "44853": ("Cabinet", "cabinet.usd"),
    "7320": ("Microwave", "microwave.usd"),
}


@dataclass(frozen=True)
class ModulePaths:
    module_root: Path
    isaaclab_root: Path
    raw_root: Path
    usd_root: Path
    isaaclab_sh: Path
    convert_urdf_py: Path


def _find_paths() -> ModulePaths:
    module_root = Path(__file__).resolve().parents[1]
    isaaclab_root = module_root.parent
    raw_root = module_root / "raw_sapien"
    usd_root = module_root / "usd_assets"
    isaaclab_sh = isaaclab_root / "isaaclab.sh"
    convert_urdf_py = isaaclab_root / "scripts" / "tools" / "convert_urdf.py"
    return ModulePaths(module_root, isaaclab_root, raw_root, usd_root, isaaclab_sh, convert_urdf_py)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _safe_filename(path: Path) -> str:
    return _safe_name(path.stem) + path.suffix.lower()


def _extract_zip_asset(zip_path: Path, paths: ModulePaths) -> Path:
    asset_id = zip_path.stem
    dst = paths.raw_root / asset_id
    if dst.exists():
        raise FileExistsError(f"Raw asset already exists: {dst}")
    tmp_dst = paths.raw_root / f".extracting_{asset_id}"
    if tmp_dst.exists():
        shutil.rmtree(tmp_dst)
    tmp_dst.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(tmp_dst)

    if (tmp_dst / DEFAULT_URDF_NAME).is_file():
        tmp_dst.rename(dst)
        return dst.resolve()

    children = [path for path in tmp_dst.iterdir() if path.is_dir()]
    if len(children) == 1 and (children[0] / DEFAULT_URDF_NAME).is_file():
        children[0].rename(dst)
        shutil.rmtree(tmp_dst)
        return dst.resolve()

    shutil.rmtree(tmp_dst)
    raise FileNotFoundError(f"Zip does not contain {DEFAULT_URDF_NAME} at its root or one top-level folder: {zip_path}")


def _resolve_asset_dir(value: str, paths: ModulePaths, allow_zip_import: bool = False) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    if allow_zip_import and candidate.is_file() and candidate.suffix.lower() == ".zip":
        return _extract_zip_asset(candidate.resolve(), paths)
    raw_candidate = paths.raw_root / value
    if raw_candidate.is_dir():
        return raw_candidate.resolve()
    raise FileNotFoundError(f"Asset directory not found: {value} or {raw_candidate}")


def _asset_id_from_dir(asset_dir: Path) -> str:
    return asset_dir.name


def _guess_name_and_usd(asset_id: str, asset_dir: Path) -> tuple[str, str]:
    if asset_id in KNOWN_ASSET_NAMES:
        return KNOWN_ASSET_NAMES[asset_id]
    meta_path = asset_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            category = str(meta.get("model_cat") or meta.get("category") or "SapienAsset")
            return _safe_name(category), f"{_safe_name(category).lower()}.usd"
        except json.JSONDecodeError:
            pass
    return "SapienAsset", "asset.usd"


def _rewrite_obj(text: str) -> str:
    text = re.sub(
        r"^mtllib\s+(.+)$",
        lambda match: f"mtllib {_safe_filename(Path(match.group(1).strip()))}",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^usemtl\s+(.+)$",
        lambda match: f"usemtl {_safe_name(match.group(1).strip())}",
        text,
        flags=re.MULTILINE,
    )
    return text


def _rewrite_mtl(text: str) -> str:
    return re.sub(
        r"^newmtl\s+(.+)$",
        lambda match: f"newmtl {_safe_name(match.group(1).strip())}",
        text,
        flags=re.MULTILINE,
    )


def prepare_urdf(asset_dir: Path, urdf_name: str = DEFAULT_URDF_NAME, output_name: str = DEFAULT_PREPARED_URDF_NAME) -> Path:
    """Create sanitized mesh copies and an Isaac-ready URDF in ``asset_dir``."""

    asset_dir = asset_dir.resolve()
    urdf_path = asset_dir / urdf_name
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    mesh_dir = asset_dir / "textured_objs"
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")

    isaac_mesh_dir = asset_dir / "textured_objs_isaac"
    if isaac_mesh_dir.exists():
        shutil.rmtree(isaac_mesh_dir)
    isaac_mesh_dir.mkdir(parents=True)

    replacements: dict[str, str] = {}
    for src_path in sorted(mesh_dir.iterdir()):
        if not src_path.is_file():
            continue
        dst_name = _safe_filename(src_path)
        dst_path = isaac_mesh_dir / dst_name
        if src_path.suffix.lower() == ".obj":
            dst_path.write_text(_rewrite_obj(src_path.read_text(encoding="utf-8")), encoding="utf-8")
        elif src_path.suffix.lower() == ".mtl":
            dst_path.write_text(_rewrite_mtl(src_path.read_text(encoding="utf-8")), encoding="utf-8")
        else:
            shutil.copy2(src_path, dst_path)
        replacements[f"textured_objs/{src_path.name}"] = f"textured_objs_isaac/{dst_name}"

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for elem in root.iter():
        if "name" in elem.attrib:
            elem.attrib["name"] = _safe_name(elem.attrib["name"])

    removed_meshes: list[str] = []
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for geom_elem in list(link.findall(tag)):
                mesh = geom_elem.find("./geometry/mesh")
                if mesh is None:
                    continue
                src = mesh.attrib.get("filename")
                if not src:
                    continue
                if src in replacements:
                    mesh.attrib["filename"] = replacements[src]
                elif not (asset_dir / src).is_file():
                    link.remove(geom_elem)
                    removed_meshes.append(src)

    output_path = asset_dir / output_name
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"[prepare] wrote {output_path}")
    if removed_meshes:
        print("[prepare] removed URDF visual/collision elements with missing meshes:")
        for mesh in sorted(set(removed_meshes)):
            print(f"  {mesh}")
    return output_path


def import_raw_asset(src: Path, asset_id: str | None, paths: ModulePaths) -> Path:
    src = src.expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src}")
    dst_name = asset_id or src.name
    dst = paths.raw_root / dst_name
    if dst.exists():
        raise FileExistsError(f"Raw asset already exists: {dst}")
    shutil.copytree(src, dst)
    print(f"[import] copied {src} -> {dst}")
    return dst


def convert_asset(args: argparse.Namespace, paths: ModulePaths) -> Path:
    asset_dir = _resolve_asset_dir(args.asset, paths, allow_zip_import=True)
    asset_id = _asset_id_from_dir(asset_dir)
    guessed_name, guessed_usd = _guess_name_and_usd(asset_id, asset_dir)
    asset_name = _safe_name(args.name or guessed_name)
    usd_name = args.usd_name or guessed_usd
    if not usd_name.endswith(".usd"):
        usd_name += ".usd"

    prepared_urdf = prepare_urdf(asset_dir, args.urdf_name, args.prepared_urdf_name)
    out_dir = paths.usd_root / f"{asset_name}_{asset_id}"
    out_usd = out_dir / usd_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(paths.isaaclab_sh),
        "-p",
        str(paths.convert_urdf_py.relative_to(paths.isaaclab_root)),
        str(prepared_urdf),
        str(out_usd),
        "--joint-stiffness",
        str(args.joint_stiffness),
        "--joint-damping",
        str(args.joint_damping),
        "--joint-target-type",
        args.joint_target_type,
    ]
    if args.fix_base:
        cmd.append("--fix-base")
    if args.merge_joints:
        cmd.append("--merge-joints")
    if args.headless:
        cmd.append("--headless")

    print("[convert] command:")
    print(" ".join(cmd))
    if args.dry_run:
        return out_usd

    subprocess.run(cmd, cwd=paths.isaaclab_root, check=True)
    if not out_usd.is_file():
        raise RuntimeError(f"USD conversion finished but output was not found: {out_usd}")
    print(f"[convert] wrote {out_usd}")
    return out_usd


def list_assets(paths: ModulePaths) -> None:
    print(f"raw_sapien: {paths.raw_root}")
    for path in sorted(paths.raw_root.iterdir()) if paths.raw_root.is_dir() else []:
        if path.is_dir():
            print(f"  {path.name}")
    print(f"usd_assets: {paths.usd_root}")
    for path in sorted(paths.usd_root.iterdir()) if paths.usd_root.is_dir() else []:
        if path.is_dir():
            usd_files = ", ".join(p.name for p in sorted(path.glob("*.usd"))) or "no root usd"
            print(f"  {path.name}: {usd_files}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List raw SAPIEN assets and converted USD asset folders.")

    import_parser = subparsers.add_parser("import", help="Copy a downloaded SAPIEN asset into raw_sapien/.")
    import_parser.add_argument("src", type=Path, help="Downloaded asset directory containing mobility.urdf.")
    import_parser.add_argument("--asset-id", help="Destination folder name under raw_sapien/. Defaults to src basename.")

    prepare_parser = subparsers.add_parser("prepare", help="Only sanitize URDF and mesh filenames.")
    prepare_parser.add_argument("asset", help="Asset id under raw_sapien/ or a direct asset directory path.")
    prepare_parser.add_argument("--urdf-name", default=DEFAULT_URDF_NAME)
    prepare_parser.add_argument("--prepared-urdf-name", default=DEFAULT_PREPARED_URDF_NAME)

    convert_parser = subparsers.add_parser("convert", help="Sanitize and convert one SAPIEN asset to USD.")
    convert_parser.add_argument("asset", help="Asset id under raw_sapien/, a direct asset directory path, or a .zip file.")
    convert_parser.add_argument("--name", help="Readable object name for usd_assets/<Name_asset_id>/")
    convert_parser.add_argument("--usd-name", help="USD filename. Defaults to known name or asset.usd.")
    convert_parser.add_argument("--urdf-name", default=DEFAULT_URDF_NAME)
    convert_parser.add_argument("--prepared-urdf-name", default=DEFAULT_PREPARED_URDF_NAME)
    convert_parser.add_argument("--no-fix-base", dest="fix_base", action="store_false", help="Do not fix the base link.")
    convert_parser.add_argument("--merge-joints", action="store_true", help="Merge fixed joints in the URDF importer.")
    convert_parser.add_argument("--joint-stiffness", type=float, default=0.0)
    convert_parser.add_argument("--joint-damping", type=float, default=3.0)
    convert_parser.add_argument("--joint-target-type", choices=["position", "velocity", "none"], default="none")
    convert_parser.add_argument("--no-headless", dest="headless", action="store_false", help="Open the converted stage.")
    convert_parser.add_argument("--dry-run", action="store_true", help="Prepare the URDF and print the conversion command.")
    convert_parser.set_defaults(fix_base=True, headless=True)

    return parser


def main() -> int:
    paths = _find_paths()
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "list":
        list_assets(paths)
    elif args.command == "import":
        import_raw_asset(args.src, args.asset_id, paths)
    elif args.command == "prepare":
        prepare_urdf(_resolve_asset_dir(args.asset, paths), args.urdf_name, args.prepared_urdf_name)
    elif args.command == "convert":
        if not paths.isaaclab_sh.is_file():
            raise FileNotFoundError(f"Missing isaaclab.sh: {paths.isaaclab_sh}")
        if not paths.convert_urdf_py.is_file():
            raise FileNotFoundError(f"Missing convert_urdf.py: {paths.convert_urdf_py}")
        convert_asset(args, paths)
    else:
        parser.error(f"Unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
