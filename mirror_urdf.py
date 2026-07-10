#!/usr/bin/env python3
"""
mirror_urdf.py - Mirror a URDF robot across the YZ plane (flip the X axis)
while keeping all joints functional.

Transformation rules applied (reflection M = diag(-1, 1, 1)):
  * origin xyz:  (x, y, z)    -> (-x,  y,  z)
  * origin rpy:  (r, p, y)    -> ( r, -p, -y)
  * revolute / continuous axis (pseudovector): (ax, ay, az) -> (ax, -ay, -az)
      -> joint limits and joint angles are preserved
  * prismatic axis (plain vector):             (ax, ay, az) -> (-ax, ay, az)
      -> joint limits preserved
  * inertia products: ixy -> -ixy, ixz -> -ixz (iyz, diagonal unchanged)
  * mesh files: geometry mirrored across X in the mesh's own frame,
      triangle winding fixed automatically (via trimesh)
  * optional: swap 'left'/'right' (and 'l_'/'r_' style prefixes) in names

Usage:
  python mirror_urdf.py robot.urdf mirrored_robot.urdf [--swap-names] [--mesh-suffix _mirrored]

Mirrored mesh files are written next to the originals with a suffix
(default '_mirrored') and the URDF is updated to reference them.
"""

import argparse
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

try:
    import trimesh
    HAVE_TRIMESH = True
except ImportError:
    HAVE_TRIMESH = False


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def parse_floats(s, n=None):
    vals = [float(v) for v in s.replace(",", " ").split()]
    if n is not None and len(vals) != n:
        raise ValueError(f"Expected {n} values, got {len(vals)}: '{s}'")
    return vals


def fmt(vals):
    return " ".join(f"{v:.12g}" for v in vals)


def mirror_origin(elem):
    """Apply xyz (x->-x) and rpy (r, -p, -y) rules to an <origin> element."""
    if elem is None:
        return
    if "xyz" in elem.attrib:
        x, y, z = parse_floats(elem.get("xyz"), 3)
        elem.set("xyz", fmt([-x, y, z]))
    if "rpy" in elem.attrib:
        r, p, yw = parse_floats(elem.get("rpy"), 3)
        elem.set("rpy", fmt([r, -p, -yw]))


# ----------------------------------------------------------------------------
# Mesh handling
# ----------------------------------------------------------------------------

def resolve_mesh_path(filename, urdf_dir):
    """Resolve package:// or relative mesh paths to a real file path.

    Returns (path or None, prefix) where prefix is what to keep in front of
    the rewritten filename in the URDF.
    """
    if filename.startswith("package://"):
        # package://pkg_name/rest/of/path -- try to find it relative to the
        # URDF directory by walking up and matching the tail of the path.
        rel = urlparse(filename).path.lstrip("/")          # rest/of/path
        pkg = urlparse(filename).netloc                    # pkg_name
        candidates = [
            urdf_dir / rel,
            urdf_dir / pkg / rel,
            urdf_dir.parent / pkg / rel,
            urdf_dir.parent / rel,
        ]
        for c in candidates:
            if c.exists():
                return c, filename[: len(filename) - len(rel)]
        return None, ""
    elif filename.startswith("file://"):
        p = Path(urlparse(filename).path)
        return (p if p.exists() else None), "file://"
    else:
        p = (urdf_dir / filename).resolve()
        return (p if p.exists() else None), ""


def mirror_mesh_file(src: Path, suffix: str) -> Path | None:
    """Mirror a mesh across its local X axis and save alongside the original."""
    if not HAVE_TRIMESH:
        return None
    dst = src.with_name(src.stem + suffix + src.suffix)
    if dst.exists():
        return dst  # already mirrored earlier in this run (shared mesh)
    try:
        loaded = trimesh.load(str(src), force="mesh", process=False)
        M = [[-1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
        # trimesh detects the negative determinant and re-inverts the face
        # winding so normals stay outward-facing.
        loaded.apply_transform(M)
        loaded.export(str(dst))
        return dst
    except Exception as e:
        print(f"  WARNING: could not mirror mesh {src}: {e}", file=sys.stderr)
        return None


def process_geometry(geom_parent, urdf_dir, suffix, mirrored_log):
    """Handle the <geometry> inside a visual/collision element."""
    geometry = geom_parent.find("geometry")
    if geometry is None:
        return
    mesh = geometry.find("mesh")
    if mesh is None:
        # box / cylinder / sphere are symmetric about their own frame;
        # the origin transform already handles them.
        return

    filename = mesh.get("filename", "")
    src, prefix = resolve_mesh_path(filename, urdf_dir)

    if src is not None and HAVE_TRIMESH:
        dst = mirror_mesh_file(src, suffix)
        if dst is not None:
            tail = filename[len(prefix):] if prefix else filename
            new_tail = str(Path(tail).with_name(dst.name)) if prefix or not Path(tail).is_absolute() else str(dst)
            mesh.set("filename", prefix + new_tail.replace("\\", "/"))
            mirrored_log.add(str(src))
            return

    # Fallback: negative scale in the URDF itself. Works in some viewers
    # (RViz), unreliable in physics engines -- warn the user.
    sc = parse_floats(mesh.get("scale", "1 1 1"), 3)
    mesh.set("scale", fmt([-sc[0], sc[1], sc[2]]))
    print(f"  WARNING: mesh '{filename}' not found or trimesh unavailable; "
          f"used scale='-1' fallback (may break collision in physics engines).",
          file=sys.stderr)


# ----------------------------------------------------------------------------
# Name swapping
# ----------------------------------------------------------------------------

SWAP_PATTERNS = [
    (re.compile(r"\bleft\b", re.I), "right"),
    (re.compile(r"\bright\b", re.I), "left"),
    (re.compile(r"(^|_)l(_|$)"), r"\1R\2"),   # temp uppercase to avoid double swap
    (re.compile(r"(^|_)r(_|$)"), r"\1l\2"),
]


def swap_name(name: str) -> str:
    # word-level left/right swap using a placeholder to avoid double swapping
    s = re.sub(r"left", "\x00", name, flags=re.I)
    s = re.sub(r"right", "left", s, flags=re.I)
    s = s.replace("\x00", "right")
    # l_/r_ prefix-style swap
    s = re.sub(r"(^|_)l(_|$)", lambda m: m.group(1) + "\x00" + m.group(2), s)
    s = re.sub(r"(^|_)r(_|$)", lambda m: m.group(1) + "l" + m.group(2), s)
    s = s.replace("\x00", "r")
    return s


# ----------------------------------------------------------------------------
# Main mirroring pass
# ----------------------------------------------------------------------------

def mirror_urdf(in_path: Path, out_path: Path, swap_names: bool, suffix: str):
    tree = ET.parse(in_path)
    root = tree.getroot()
    urdf_dir = in_path.parent.resolve()
    mirrored_meshes = set()

    # ---- links ----
    for link in root.iter("link"):
        inertial = link.find("inertial")
        if inertial is not None:
            mirror_origin(inertial.find("origin"))
            inertia = inertial.find("inertia")
            if inertia is not None:
                for prod in ("ixy", "ixz"):
                    if prod in inertia.attrib:
                        inertia.set(prod, fmt([-float(inertia.get(prod))]))

        for tag in ("visual", "collision"):
            for elem in link.findall(tag):
                mirror_origin(elem.find("origin"))
                process_geometry(elem, urdf_dir, suffix, mirrored_meshes)

    # ---- joints ----
    for joint in root.iter("joint"):
        jtype = joint.get("type", "fixed")
        mirror_origin(joint.find("origin"))

        axis = joint.find("axis")
        if axis is not None and "xyz" in axis.attrib:
            ax, ay, az = parse_floats(axis.get("xyz"), 3)
            if jtype in ("revolute", "continuous"):
                # pseudovector: limits & joint angles preserved
                axis.set("xyz", fmt([ax, -ay, -az]))
            elif jtype == "prismatic":
                # plain vector: limits preserved
                axis.set("xyz", fmt([-ax, ay, az]))
            elif jtype in ("planar", "floating"):
                print(f"  WARNING: joint '{joint.get('name')}' has type "
                      f"'{jtype}'; verify its behavior manually.", file=sys.stderr)

        mimic = joint.find("mimic")
        if mimic is not None:
            # With the axis conventions above, joint values are preserved,
            # so mimic multiplier/offset stay valid. Just note it.
            print(f"  NOTE: joint '{joint.get('name')}' has a <mimic> tag; "
                  f"multiplier left unchanged (should be correct with the "
                  f"preserved-angle convention). Verify.", file=sys.stderr)

    # ---- optional name swapping (links, joints, references) ----
    if swap_names:
        for link in root.iter("link"):
            link.set("name", swap_name(link.get("name", "")))
        for joint in root.iter("joint"):
            joint.set("name", swap_name(joint.get("name", "")))
            for ref in ("parent", "child"):
                e = joint.find(ref)
                if e is not None and "link" in e.attrib:
                    e.set("link", swap_name(e.get("link")))
            mimic = joint.find("mimic")
            if mimic is not None and "joint" in mimic.attrib:
                mimic.set("joint", swap_name(mimic.get("joint")))
        # transmissions / gazebo refs, best-effort
        for e in root.iter():
            if e.tag in ("transmission",) and "name" in e.attrib:
                e.set("name", swap_name(e.get("name")))
            if e.tag == "gazebo" and "reference" in e.attrib:
                e.set("reference", swap_name(e.get("reference")))

    if "name" in root.attrib:
        root.set("name", root.get("name") + "_mirrored")

    tree.write(out_path, encoding="unicode", xml_declaration=True)
    print(f"Wrote {out_path}")
    if mirrored_meshes:
        print(f"Mirrored {len(mirrored_meshes)} mesh file(s) with suffix '{suffix}'.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="input URDF file")
    ap.add_argument("output", type=Path, help="output (mirrored) URDF file")
    ap.add_argument("--swap-names", action="store_true",
                    help="swap left/right (and l_/r_) in link/joint names")
    ap.add_argument("--mesh-suffix", default="_mirrored",
                    help="suffix for mirrored mesh files (default: _mirrored)")
    args = ap.parse_args()

    if not HAVE_TRIMESH:
        print("NOTE: trimesh not installed (pip install trimesh); "
              "falling back to negative mesh scale in the URDF.", file=sys.stderr)

    mirror_urdf(args.input, args.output, args.swap_names, args.mesh_suffix)