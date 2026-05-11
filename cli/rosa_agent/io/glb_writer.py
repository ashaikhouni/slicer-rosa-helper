"""Minimal pure-Python GLB writer for the ``export-view`` CLI command.

Why a hand-rolled writer instead of ``trimesh`` / ``pygltflib``: per
``pyproject.toml`` the headless rosa-agent install pins to numpy +
SimpleITK + nibabel + scipy and nothing else. Adding a mesh-export
dependency just for view-export would bloat the install for the 99%
of users who never run this command. The GLB binary container is
small enough to assemble directly: one JSON chunk + one BIN chunk.

Scope deliberately narrow:
  * Indexed triangle meshes (positions + indices + optional vertex
    RGBA colors).
  * Instanced spheres (we build the unit-sphere once and reuse via
    per-node TRS).
  * Line segments rendered as thin cylinders (Three.js / model-viewer
    do not render line-mode reliably across stacks; cylinders work
    everywhere).
  * Optional per-node "extras" metadata so the HTML viewer can list
    trajectory + contact names alongside the 3D picks.

References:
  * glTF 2.0 spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
  * GLB container: §3.5
"""

from __future__ import annotations

import base64
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


GLB_MAGIC = 0x46546C67  # 'glTF'
GLB_VERSION = 2
CHUNK_JSON = 0x4E4F534A  # 'JSON'
CHUNK_BIN = 0x004E4942   # 'BIN\0'

GL_FLOAT = 5126
GL_UNSIGNED_INT = 5125
GL_UNSIGNED_BYTE = 5121

TARGET_ARRAY_BUFFER = 34962
TARGET_ELEMENT_ARRAY_BUFFER = 34963


# ---------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------


@dataclass
class _MeshPrimitive:
    positions: np.ndarray             # (N, 3) float32
    indices: np.ndarray               # (M*3,) uint32
    colors_rgba: np.ndarray | None    # (N, 4) uint8 or None
    material_index: int
    normals: np.ndarray | None = None  # (N, 3) float32; optional smooth normals


@dataclass
class _NodeRecord:
    name: str
    mesh_index: int
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # quaternion xyzw
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    extras: dict[str, Any] | None = None


@dataclass
class _Material:
    name: str
    base_color: tuple[float, float, float, float]
    metallic: float = 0.1
    roughness: float = 0.7
    double_sided: bool = True
    alpha_mode: str = "OPAQUE"  # "OPAQUE" | "MASK" | "BLEND"


@dataclass
class GLBScene:
    """Buffer of primitives + nodes waiting for ``encode_glb`` to pack them."""

    primitives: list[_MeshPrimitive] = field(default_factory=list)
    nodes: list[_NodeRecord] = field(default_factory=list)
    materials: list[_Material] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)
    _sphere_primitive: int | None = None
    _cylinder_primitive: int | None = None

    # ---- materials --------------------------------------------------

    def add_material(
        self,
        name: str,
        rgba: Sequence[float],
        *,
        metallic: float = 0.1,
        roughness: float = 0.7,
        double_sided: bool = True,
        alpha_mode: str = "OPAQUE",
    ) -> int:
        self.materials.append(_Material(
            name=name,
            base_color=(float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])),
            metallic=metallic, roughness=roughness, double_sided=double_sided,
            alpha_mode=alpha_mode,
        ))
        return len(self.materials) - 1

    # ---- primitives -------------------------------------------------

    def _add_primitive(
        self,
        positions: np.ndarray,
        indices: np.ndarray,
        material_index: int,
        colors_rgba: np.ndarray | None = None,
        normals: np.ndarray | None = None,
    ) -> int:
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must be (N,3); got {positions.shape}")
        indices = np.ascontiguousarray(indices, dtype=np.uint32).reshape(-1)
        if indices.size % 3:
            raise ValueError(f"indices length {indices.size} is not a multiple of 3")
        if colors_rgba is not None:
            colors_rgba = np.ascontiguousarray(colors_rgba, dtype=np.uint8)
            if colors_rgba.shape != (positions.shape[0], 4):
                raise ValueError(
                    f"colors_rgba shape {colors_rgba.shape} doesn't match "
                    f"positions ({positions.shape[0]} verts)"
                )
        if normals is not None:
            normals = np.ascontiguousarray(normals, dtype=np.float32)
            if normals.shape != positions.shape:
                raise ValueError(
                    f"normals shape {normals.shape} doesn't match "
                    f"positions {positions.shape}"
                )
        self.primitives.append(_MeshPrimitive(
            positions=positions, indices=indices,
            colors_rgba=colors_rgba, material_index=material_index,
            normals=normals,
        ))
        return len(self.primitives) - 1

    # ---- public adders ---------------------------------------------

    def add_surface(
        self,
        name: str,
        positions: np.ndarray,
        faces: np.ndarray,
        material_index: int,
        *,
        vertex_colors_rgba: np.ndarray | None = None,
        normals: np.ndarray | None = None,
        extras: dict[str, Any] | None = None,
    ) -> int:
        """One indexed triangle mesh -> one mesh -> one node."""
        prim = self._add_primitive(positions, faces.reshape(-1), material_index,
                                   colors_rgba=vertex_colors_rgba, normals=normals)
        self.nodes.append(_NodeRecord(name=name, mesh_index=prim, extras=extras))
        return len(self.nodes) - 1

    def _ensure_sphere(self) -> int:
        if self._sphere_primitive is not None:
            return self._sphere_primitive
        verts, faces = _generate_unit_sphere(stacks=10, slices=14)
        # Material assigned per-node via instancing — but glTF binds
        # material at the primitive level, not the node level. To allow
        # multiple materials, we duplicate the mesh primitive per
        # material. For now we use a default "contact" material that
        # callers can override by adding a per-contact material and
        # passing a material_index to ``add_sphere``.
        mat = self.add_material("__sphere_placeholder", (1.0, 1.0, 1.0, 1.0))
        prim = self._add_primitive(verts, faces.reshape(-1), mat)
        self._sphere_primitive = prim
        return prim

    def _ensure_cylinder(self) -> int:
        if self._cylinder_primitive is not None:
            return self._cylinder_primitive
        verts, faces = _generate_unit_cylinder(segments=16)
        mat = self.add_material("__cylinder_placeholder", (1.0, 1.0, 1.0, 1.0))
        prim = self._add_primitive(verts, faces.reshape(-1), mat)
        self._cylinder_primitive = prim
        return prim

    def add_sphere(
        self,
        name: str,
        center: Sequence[float],
        radius: float,
        material_index: int,
        *,
        extras: dict[str, Any] | None = None,
    ) -> int:
        """Add an instanced unit-sphere node, scaled by ``radius``.

        The sphere primitive is created on first call and re-used; each
        contact only adds one Node referencing the shared mesh, so 200
        contacts cost ~200 nodes (cheap) instead of 200 meshes.
        """
        prim = self._duplicate_primitive_for_material(self._ensure_sphere(), material_index)
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        r = float(radius)
        self.nodes.append(_NodeRecord(
            name=name, mesh_index=prim,
            translation=(cx, cy, cz), scale=(r, r, r),
            extras=extras,
        ))
        return len(self.nodes) - 1

    def add_segment(
        self,
        name: str,
        p0: Sequence[float],
        p1: Sequence[float],
        radius: float,
        material_index: int,
        *,
        extras: dict[str, Any] | None = None,
    ) -> int:
        """Draw the segment ``p0 -> p1`` as a thin cylinder."""
        prim = self._duplicate_primitive_for_material(self._ensure_cylinder(), material_index)
        a = np.asarray(p0, dtype=float)
        b = np.asarray(p1, dtype=float)
        midpoint = (a + b) * 0.5
        direction = b - a
        length = float(np.linalg.norm(direction))
        if length <= 0.0:
            # Degenerate segment — emit a sphere instead of a zero-length cylinder.
            return self.add_sphere(name, midpoint, max(radius, 1e-3),
                                   material_index, extras=extras)
        # Unit cylinder runs from y=-0.5 to y=+0.5 with radius 1. Scale
        # X/Z by radius, Y by length, then rotate to align +Y with
        # ``direction`` and translate to midpoint.
        rot_xyzw = _quat_from_two_vectors(np.array([0.0, 1.0, 0.0]), direction / length)
        self.nodes.append(_NodeRecord(
            name=name, mesh_index=prim,
            translation=(float(midpoint[0]), float(midpoint[1]), float(midpoint[2])),
            rotation=tuple(float(v) for v in rot_xyzw),  # type: ignore[arg-type]
            scale=(float(radius), length, float(radius)),
            extras=extras,
        ))
        return len(self.nodes) - 1

    # ---- helpers ----------------------------------------------------

    def _duplicate_primitive_for_material(
        self, source_prim_index: int, material_index: int,
    ) -> int:
        """Return a primitive that uses ``material_index``.

        glTF binds material at the primitive level; instancing across
        multiple materials means duplicating the (cheap) primitive
        descriptor while sharing the underlying vertex / index buffers.
        We re-add the same numpy arrays — the buffer-packing step will
        deduplicate identical byte ranges.
        """
        src = self.primitives[source_prim_index]
        if src.material_index == material_index:
            return source_prim_index
        # Look for an existing duplicate first so 200 same-material
        # contacts share one primitive entry.
        for idx, prim in enumerate(self.primitives):
            if (prim.positions is src.positions
                    and prim.indices is src.indices
                    and prim.material_index == material_index):
                return idx
        self.primitives.append(_MeshPrimitive(
            positions=src.positions, indices=src.indices,
            colors_rgba=src.colors_rgba, material_index=material_index,
        ))
        return len(self.primitives) - 1


# ---------------------------------------------------------------------
# Unit primitives
# ---------------------------------------------------------------------


def _generate_unit_sphere(stacks: int = 12, slices: int = 16):
    """UV sphere; centered at origin, radius 1. (verts, faces)."""
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for i in range(stacks + 1):
        phi = np.pi * i / stacks
        sp, cp = float(np.sin(phi)), float(np.cos(phi))
        for j in range(slices + 1):
            theta = 2.0 * np.pi * j / slices
            st, ct = float(np.sin(theta)), float(np.cos(theta))
            verts.append((sp * ct, cp, sp * st))
    for i in range(stacks):
        for j in range(slices):
            a = i * (slices + 1) + j
            b = a + slices + 1
            # CCW from outside: (a, a+1, b) and (a+1, b+1, b).
            faces.append((a, a + 1, b))
            faces.append((a + 1, b + 1, b))
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.uint32)


def _generate_unit_cylinder(segments: int = 16):
    """Cylinder along +Y, radius 1, height 1 centered at origin. (verts, faces)."""
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    # Side ring vertices: top + bottom for each segment column (allows
    # vertex doubling-free triangulation around the seam).
    for s in range(segments + 1):
        theta = 2.0 * np.pi * s / segments
        x, z = float(np.cos(theta)), float(np.sin(theta))
        verts.append((x, -0.5, z))   # bottom
        verts.append((x, +0.5, z))   # top
    # Sides. CCW from outside: (b0, t0, t1) and (b0, t1, b1).
    for s in range(segments):
        b0 = 2 * s
        t0 = 2 * s + 1
        b1 = 2 * (s + 1)
        t1 = 2 * (s + 1) + 1
        faces.append((b0, t0, t1))
        faces.append((b0, t1, b1))
    # Caps. Top normal = +Y (CCW viewed from +Y); bottom = -Y.
    top_center = len(verts); verts.append((0.0, +0.5, 0.0))
    bot_center = len(verts); verts.append((0.0, -0.5, 0.0))
    for s in range(segments):
        t0 = 2 * s + 1
        t1 = 2 * (s + 1) + 1
        b0 = 2 * s
        b1 = 2 * (s + 1)
        faces.append((top_center, t0, t1))
        faces.append((bot_center, b1, b0))
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.uint32)


def _quat_from_two_vectors(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Quaternion xyzw rotating ``u`` to ``v`` (both unit vectors)."""
    u = u / max(np.linalg.norm(u), 1e-12)
    v = v / max(np.linalg.norm(v), 1e-12)
    d = float(np.dot(u, v))
    if d > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if d < -0.999999:
        # Pick any orthogonal axis to flip 180°.
        axis = np.cross(u, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(u, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])
    cross = np.cross(u, v)
    w = 1.0 + d
    q = np.array([cross[0], cross[1], cross[2], w])
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------


def _pack_buffer(scene: GLBScene) -> tuple[bytes, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Pack all primitive arrays into one BIN blob; return blob + glTF tables."""
    blob = bytearray()
    buffer_views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    primitive_records: list[dict[str, Any]] = []
    cache: dict[int, dict[str, int]] = {}  # id(array) -> {position, indices, colors} accessor indices

    def _pad_to_4():
        while len(blob) % 4:
            blob.append(0)

    def _add_view(data: bytes, target: int | None) -> int:
        _pad_to_4()
        offset = len(blob)
        blob.extend(data)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        buffer_views.append(view)
        return len(buffer_views) - 1

    def _add_accessor_floats(arr: np.ndarray, kind: str) -> int:
        data = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
        view = _add_view(data, TARGET_ARRAY_BUFFER)
        mins = arr.min(axis=0).tolist() if arr.size else [0.0] * arr.shape[1]
        maxs = arr.max(axis=0).tolist() if arr.size else [0.0] * arr.shape[1]
        accessors.append({
            "bufferView": view,
            "componentType": GL_FLOAT,
            "count": int(arr.shape[0]),
            "type": kind,
            "min": [float(v) for v in mins],
            "max": [float(v) for v in maxs],
        })
        return len(accessors) - 1

    def _add_accessor_indices(arr: np.ndarray) -> int:
        data = np.ascontiguousarray(arr, dtype=np.uint32).tobytes()
        view = _add_view(data, TARGET_ELEMENT_ARRAY_BUFFER)
        accessors.append({
            "bufferView": view,
            "componentType": GL_UNSIGNED_INT,
            "count": int(arr.size),
            "type": "SCALAR",
        })
        return len(accessors) - 1

    def _add_accessor_colors(arr: np.ndarray) -> int:
        # Normalized uint8 RGBA -> COLOR_0.
        data = np.ascontiguousarray(arr, dtype=np.uint8).tobytes()
        view = _add_view(data, TARGET_ARRAY_BUFFER)
        accessors.append({
            "bufferView": view,
            "componentType": GL_UNSIGNED_BYTE,
            "count": int(arr.shape[0]),
            "type": "VEC4",
            "normalized": True,
        })
        return len(accessors) - 1

    for prim in scene.primitives:
        key = (id(prim.positions), id(prim.indices),
               id(prim.colors_rgba) if prim.colors_rgba is not None else 0,
               id(prim.normals) if prim.normals is not None else 0)
        slot = cache.get(key)
        if slot is None:
            slot = {
                "POSITION": _add_accessor_floats(prim.positions, "VEC3"),
                "INDICES": _add_accessor_indices(prim.indices),
            }
            if prim.colors_rgba is not None:
                slot["COLOR_0"] = _add_accessor_colors(prim.colors_rgba)
            if prim.normals is not None:
                slot["NORMAL"] = _add_accessor_floats(prim.normals, "VEC3")
            cache[key] = slot
        attrs = {"POSITION": slot["POSITION"]}
        if "COLOR_0" in slot:
            attrs["COLOR_0"] = slot["COLOR_0"]
        if "NORMAL" in slot:
            attrs["NORMAL"] = slot["NORMAL"]
        primitive_records.append({
            "attributes": attrs,
            "indices": slot["INDICES"],
            "material": prim.material_index,
        })

    _pad_to_4()
    return bytes(blob), buffer_views, accessors, primitive_records


def encode_glb(scene: GLBScene) -> bytes:
    """Pack a ``GLBScene`` into a ``.glb`` binary blob (single bytes object)."""
    bin_blob, buffer_views, accessors, primitive_records = _pack_buffer(scene)

    # Each scene.primitives entry becomes its own mesh + node. Material
    # references are already encoded inside primitive_records. We pack
    # one mesh per node so the node table is the authoritative naming
    # surface for downstream picking.
    meshes_table: list[dict[str, Any]] = []
    nodes_table: list[dict[str, Any]] = []
    for node in scene.nodes:
        prim_rec = primitive_records[node.mesh_index]
        mesh_index = len(meshes_table)
        meshes_table.append({
            "name": f"{node.name}_mesh",
            "primitives": [prim_rec],
        })
        node_entry: dict[str, Any] = {
            "name": node.name,
            "mesh": mesh_index,
        }
        if node.translation != (0.0, 0.0, 0.0):
            node_entry["translation"] = list(node.translation)
        if node.rotation != (0.0, 0.0, 0.0, 1.0):
            node_entry["rotation"] = list(node.rotation)
        if node.scale != (1.0, 1.0, 1.0):
            node_entry["scale"] = list(node.scale)
        if node.extras:
            node_entry["extras"] = node.extras
        nodes_table.append(node_entry)

    materials_table = []
    for mat in scene.materials:
        entry: dict[str, Any] = {
            "name": mat.name,
            "pbrMetallicRoughness": {
                "baseColorFactor": list(mat.base_color),
                "metallicFactor": mat.metallic,
                "roughnessFactor": mat.roughness,
            },
            "doubleSided": mat.double_sided,
        }
        if mat.alpha_mode and mat.alpha_mode != "OPAQUE":
            entry["alphaMode"] = mat.alpha_mode
        materials_table.append(entry)

    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "rosa-agent export-view"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes_table)))}],
        "nodes": nodes_table,
        "meshes": meshes_table,
        "materials": materials_table,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(bin_blob)}],
    }
    if scene.extras:
        gltf["extras"] = scene.extras

    json_blob = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(json_blob) % 4:
        json_blob += b" "
    while len(bin_blob) % 4:
        bin_blob += b"\x00"

    total_length = 12 + 8 + len(json_blob) + 8 + len(bin_blob)
    header = struct.pack("<III", GLB_MAGIC, GLB_VERSION, total_length)
    json_chunk = struct.pack("<II", len(json_blob), CHUNK_JSON) + json_blob
    bin_chunk = struct.pack("<II", len(bin_blob), CHUNK_BIN) + bin_blob
    return header + json_chunk + bin_chunk


def write_glb(path: str | Path, scene: GLBScene) -> int:
    """Encode + write the scene; return written byte count."""
    data = encode_glb(scene)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return len(data)


# Re-export base64 only so test/CLI helpers that want a data: URI can
# get one without re-importing.
def glb_to_data_uri(path_or_bytes: str | Path | bytes) -> str:
    if isinstance(path_or_bytes, (str, Path)):
        data = Path(path_or_bytes).read_bytes()
    else:
        data = path_or_bytes
    return "data:model/gltf-binary;base64," + base64.b64encode(data).decode("ascii")
