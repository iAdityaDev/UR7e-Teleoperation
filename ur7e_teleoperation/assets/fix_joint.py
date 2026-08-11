#!/usr/bin/env python3
"""
Freeze every revolute joint in a URDF to 'fixed', locking the skeleton
rigid in its current pose. Use this on the human model so mesh-collision
overlap between adjacent segments (spine, pelvis, etc.) can no longer
cause self-collision forces to visibly deform the body.

Usage:
    python3 freeze_joints.py kevin_ultrasound_meshcol.urdf kevin_ultrasound_frozen.urdf
    python3 freeze_joints.py kevin_ultrasound_meshcol.urdf out.urdf --keep jRightShoulder_rotx jRightShoulder_roty
"""

import argparse
import xml.etree.ElementTree as ET


def freeze_joint(joint_elem):
    """Convert a revolute/continuous joint element to type='fixed',
    stripping the child elements that are invalid on a fixed joint."""
    joint_elem.set('type', 'fixed')

    for tag in ('axis', 'limit', 'dynamics'):
        el = joint_elem.find(tag)
        if el is not None:
            joint_elem.remove(el)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input_urdf')
    ap.add_argument('output_urdf')
    ap.add_argument('--keep', nargs='*', default=[],
                     help="Joint names to leave movable (not frozen).")
    args = ap.parse_args()

    keep = set(args.keep)

    tree = ET.parse(args.input_urdf)
    root = tree.getroot()

    frozen = 0
    kept = 0
    for joint in root.findall('joint'):
        name = joint.get('name')
        jtype = joint.get('type')

        if jtype == 'fixed':
            continue

        if name in keep:
            print(f"  [KEEP]   '{name}' ({jtype}) — left movable")
            kept += 1
            continue

        freeze_joint(joint)
        print(f"  [FROZEN] '{name}' {jtype} -> fixed")
        frozen += 1

    tree.write(args.output_urdf, encoding='utf-8', xml_declaration=True)
    print(f"\nDone. Froze {frozen} joint(s), kept {kept} movable. Wrote: {args.output_urdf}")


if __name__ == '__main__':
    main()