# Object meshes

Place your target object's **CAD mesh** here (e.g. `target_object.obj`,
`cube.obj`) and reference it from `configs/object_assets.yaml`.

Requirements for FoundationPose (model-based):

- The mesh must be a watertight-ish surface mesh (`.obj`, `.ply`, `.stl`, ...).
- **Units must be METERS.** FoundationPose assumes metric model coordinates;
  a mesh in millimeters will produce a pose off by ~1000x.
- The mesh's own coordinate frame is the `object` frame. `T_camera_object`
  returned by FoundationPose maps points from this frame into the camera frame.
- Provide normals if possible (FoundationPose uses model normals).

These mesh files are inputs you supply; they are intentionally not committed.
