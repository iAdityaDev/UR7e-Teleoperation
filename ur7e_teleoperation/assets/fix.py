#!/usr/bin/env python3
"""
Convert box-collision links to mesh-collision, using each link's primary
"skin" visual mesh (the plain {LinkName}.obj entry, not the muscle/bone
overlay meshes) as the actual collision shape.

Usage:
    python3 box_to_mesh_collision.py kevin_ultrasound.urdf kevin_ultrasound_meshcol.urdf
    python3 box_to_mesh_collision.py kevin_ultrasound.urdf out.urdf --links L3 L5 T12 Pelvis T8
"""

import os
import argparse
import xml.etree.ElementTree as ET


def find_skin_mesh(link_elem, link_name):
    """Find the visual whose mesh filename base name == link_name exactly
    (e.g. 'L5.obj' for link 'L5'), skipping muscle/bone overlays like
    'L5_LeftErcSpin.obj' or 'L5_SpinalCord.obj'."""
    for visual in link_elem.findall('visual'):
        mesh = visual.find('geometry/mesh')
        if mesh is None:
            continue
        filename = mesh.get('filename')
        base = os.path.splitext(os.path.basename(filename))[0]
        if base == link_name:
            origin = visual.find('origin')
            xyz = origin.get('xyz', '0 0 0') if origin is not None else '0 0 0'
            rpy = origin.get('rpy', '0 0 0') if origin is not None else '0 0 0'
            scale = mesh.get('scale', '1 1 1')
            return filename, scale, xyz, rpy
    return None


def convert_link(link_elem):
    name = link_elem.get('name')
    collision = link_elem.find('collision')
    if collision is None:
        print(f"  [SKIP] '{name}': no <collision>")
        return False

    box = collision.find('geometry/box')
    if box is None:
        print(f"  [SKIP] '{name}': collision is not a box, leaving alone")
        return False

    skin = find_skin_mesh(link_elem, name)
    if skin is None:
        print(f"  [SKIP] '{name}': no matching skin mesh '{name}.obj' found among visuals")
        return False

    filename, scale, xyz, rpy = skin

    geom = collision.find('geometry')
    geom.remove(box)
    mesh_elem = ET.SubElement(geom, 'mesh')
    mesh_elem.set('filename', filename)
    mesh_elem.set('scale', scale)

    origin = collision.find('origin')
    if origin is None:
        origin = ET.SubElement(collision, 'origin')
    origin.set('xyz', xyz)
    origin.set('rpy', rpy)

    print(f"  [FIXED] '{name}': box -> mesh ({os.path.basename(filename)})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input_urdf')
    ap.add_argument('output_urdf')
    ap.add_argument('--links', nargs='*', default=None)
    args = ap.parse_args()

    tree = ET.parse(args.input_urdf)
    root = tree.getroot()

    fixed = 0
    for link in root.findall('link'):
        name = link.get('name')
        if args.links is not None and name not in args.links:
            continue
        print(f"Processing '{name}'...")
        if convert_link(link):
            fixed += 1

    tree.write(args.output_urdf, encoding='utf-8', xml_declaration=True)
    print(f"\nDone. Converted {fixed} link(s) to mesh collision. Wrote: {args.output_urdf}")


if __name__ == '__main__':
    main()
    