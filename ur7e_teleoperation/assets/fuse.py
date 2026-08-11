#!/usr/bin/env python3
"""
Give Pelvis one fused collision mesh (covering Pelvis+L5+L3+T12+T8),
and strip <collision> from L5, L3, T12, T8 so nothing overlaps.
Visuals are untouched -- the body still looks the same, only collision
geometry changes.

Usage:
    python3 fuse_torso_urdf.py kevin_ultrasound_frozen.urdf kevin_ultrasound_fused.urdf
"""

import argparse
import xml.etree.ElementTree as ET

FUSED_MESH = '/home/deviant/IIIT_intern/src/ur7e_teleoperation/assets/meshes/Torso_fused_collision.obj'
LINKS_TO_STRIP = ['L5', 'L3', 'T12', 'T8']  # keep Pelvis's own collision, replace it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input_urdf')
    ap.add_argument('output_urdf')
    args = ap.parse_args()

    tree = ET.parse(args.input_urdf)
    root = tree.getroot()

    for link in root.findall('link'):
        name = link.get('name')

        if name == 'Pelvis':
            collision = link.find('collision')
            geom = collision.find('geometry')
            mesh = geom.find('mesh')
            mesh.set('filename', FUSED_MESH)
            mesh.set('scale', '1. 1. 1.')
            origin = collision.find('origin')
            origin.set('xyz', '0.0 0.0 0.0')
            origin.set('rpy', '0.0 0.0 0.0')
            print(f"  [FUSED]   'Pelvis': collision -> Torso_fused_collision.obj")

        elif name in LINKS_TO_STRIP:
            collision = link.find('collision')
            if collision is not None:
                link.remove(collision)
                print(f"  [STRIPPED] '{name}': collision removed (now covered by fused Pelvis mesh)")

    tree.write(args.output_urdf, encoding='utf-8', xml_declaration=True)
    print(f"\nDone. Wrote: {args.output_urdf}")


if __name__ == '__main__':
    main()