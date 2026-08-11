#!/usr/bin/env python3
"""
Merge Pelvis/L5/L3/T12/T8 skin meshes into ONE combined collision mesh,
expressed in Pelvis-local space, using the known joint-chain offsets from
the URDF (all these joints are pure Z-translations with zero rotation at
rest, so the transform is just cumulative Z addition).

Requires: pip install trimesh --break-system-packages

Usage:
    python3 merge_torso_collision.py
"""

import os
import trimesh
import numpy as np

MESH_DIR = '/home/deviant/IIIT_intern/src/ur7e_teleoperation/assets/meshes/'
OUT_PATH = os.path.join(MESH_DIR, 'Torso_fused_collision.obj')

# (link_skin_obj, own_local_mesh_origin_z, cumulative_joint_z_to_this_link's_frame)
# cumulative Z = sum of joint origin z's along the chain from Pelvis to this link
# (computed from the URDF's joint <origin xyz="0 0 z">, all roty joints have 0 offset)
LINKS = [
    ('Pelvis.obj', 0.0,         0.0),
    ('L5.obj',     0.02455875,  0.064935),
    ('L3.obj',     0.0374625,   0.1140525),
    ('T12.obj',    0.0437895,   0.1889775),
    ('T8.obj',     0.06901425,  0.2765565),
]

meshes = []
for filename, local_origin_z, cumulative_z in LINKS:
    path = os.path.join(MESH_DIR, filename)
    m = trimesh.load(path, process=False)
    # place mesh in its own link-local frame (mesh origin offset), then
    # shift into the shared Pelvis-frame by the cumulative joint offset
    total_z = local_origin_z + cumulative_z
    m.apply_translation([0.0, 0.0, total_z])
    meshes.append(m)
    print(f"  merged {filename}: local_origin_z={local_origin_z}, "
          f"cumulative_joint_z={cumulative_z}, total_z_shift={total_z}")

fused = trimesh.util.concatenate(meshes)
fused.export(OUT_PATH)
print(f"\nWrote fused torso collision mesh: {OUT_PATH}")
print(f"  vertices: {len(fused.vertices)}, faces: {len(fused.faces)}")