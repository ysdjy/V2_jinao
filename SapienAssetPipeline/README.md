# SAPIEN Asset Pipeline

Self-contained SAPIEN/PartNet-Mobility asset workspace for Isaac Lab.

## Layout

- `raw_sapien/`: original assets downloaded from <https://sapien.ucsd.edu/browse>.
- `usd_assets/`: converted Isaac/Omniverse USD assets.
- `tools/convert_sapien_asset.py`: one-command prepare and URDF-to-USD conversion tool.

The module is path-relative. If this folder is copied with an Isaac Lab checkout, the tool finds `isaaclab.sh` and `scripts/tools/convert_urdf.py` from the parent directory.

## Existing Assets

Raw assets copied from `Connection/USD`:

- `raw_sapien/101054`
- `raw_sapien/12252`
- `raw_sapien/44853`
- `raw_sapien/7320`

Converted USD assets copied from `Connection/assets/Props`:

- `usd_assets/Knife_101054/knife.usd`
- `usd_assets/Fridge_12252/fridge.usd`
- `usd_assets/Cabinet_44853/cabinet.usd`
- `usd_assets/Microwave_7320/microwave.usd`

## Convert A New Download

If the download is a zip file, convert it directly:

```bash
python SapienAssetPipeline/tools/convert_sapien_asset.py convert /path/to/12345.zip --name Chair
```

The zip will be extracted to `raw_sapien/12345` and converted to `usd_assets/Chair_12345/`.

If the download has already been unzipped, put the folder under `raw_sapien/<asset_id>` so that it contains `mobility.urdf` and `textured_objs/`.

Then run one command from the IsaacLab root:

```bash
python SapienAssetPipeline/tools/convert_sapien_asset.py convert <asset_id> --name <ObjectName>
```

Example:

```bash
python SapienAssetPipeline/tools/convert_sapien_asset.py convert 12345 --name Chair
```

The output goes to:

```text
SapienAssetPipeline/usd_assets/Chair_12345/
```

If the source asset is elsewhere, first import it:

```bash
python SapienAssetPipeline/tools/convert_sapien_asset.py import /path/to/downloaded/12345 --asset-id 12345
python SapienAssetPipeline/tools/convert_sapien_asset.py convert 12345 --name Chair
```

## Useful Options

Preview the conversion command without launching Isaac Sim:

```bash
python SapienAssetPipeline/tools/convert_sapien_asset.py convert 101054 --dry-run
```

Only sanitize the URDF and mesh names:

```bash
python SapienAssetPipeline/tools/convert_sapien_asset.py prepare 101054
```

Merge fixed joints during USD import:

```bash
python SapienAssetPipeline/tools/convert_sapien_asset.py convert 101054 --merge-joints
```

List known assets:

```bash
python SapienAssetPipeline/tools/convert_sapien_asset.py list
```
