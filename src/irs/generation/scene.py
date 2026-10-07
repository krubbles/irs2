"""Export DeepMIMO building and terrain meshes for Sionna ray tracing."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import numpy.typing as npt
from deepmimo.core.scene import PhysicalElement
from deepmimo.core.scene import Scene as DeepMIMOScene

FloatArray = npt.NDArray[np.float32]


def create_sionna_full_scene_xml(
    deepmimo_scene: DeepMIMOScene,
    *,
    output_directory: Path,
) -> str:
    """Export every building and terrain object for a full-scene preview."""
    output_directory.mkdir(parents=True, exist_ok=True)
    shape_xml: list[str] = []
    label_counts: dict[str, int] = {}
    for scene_object in deepmimo_scene.objects:
        if scene_object.label not in {"buildings", "terrain"}:
            continue
        object_index: int = label_counts.get(scene_object.label, 0)
        label_counts[scene_object.label] = object_index + 1
        object_name: str = f"{scene_object.label}_{object_index:03d}"
        material_id: str
        if scene_object.label == "buildings":
            material_id = "preview_concrete"
        elif "terrain" in scene_object.name.lower():
            material_id = "preview_ground"
        else:
            material_id = "preview_road"

        ply_path: Path = output_directory / f"{object_name}.ply"
        _write_ascii_ply(scene_object, ply_path)
        escaped_path: str = escape(str(ply_path))
        shape_xml.append(
            f"""
    <shape type="ply" id="{object_name}">
        <string name="filename" value="{escaped_path}"/>
        <ref id="{material_id}" name="bsdf"/>
    </shape>"""
        )

    if not shape_xml:
        raise ValueError("The DeepMIMO scene contains no building or terrain objects.")

    return f"""<scene version="2.1.0">
    <bsdf type="itu-radio-material" id="preview_concrete">
        <string name="type" value="concrete"/>
        <rgb name="color" value="0.54 0.54 0.54"/>
    </bsdf>
    <bsdf type="itu-radio-material" id="preview_ground">
        <string name="type" value="very_dry_ground"/>
        <rgb name="color" value="0.38 0.43 0.30"/>
    </bsdf>
    <bsdf type="itu-radio-material" id="preview_road">
        <string name="type" value="concrete"/>
        <rgb name="color" value="0.20 0.22 0.24"/>
    </bsdf>
    {"".join(shape_xml)}
</scene>"""


def _write_ascii_ply(scene_object: PhysicalElement, output_path: Path) -> None:
    """Write one DeepMIMO physical element as a triangular ASCII PLY mesh."""
    vertex_indices: dict[tuple[float, float, float], int] = {}
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []

    for face in scene_object.faces:
        for triangle in face.triangular_faces:
            triangle_array: FloatArray = np.asarray(triangle, dtype=np.float32)
            indices: list[int] = []
            for vertex in triangle_array:
                vertex_key: tuple[float, float, float] = (
                    float(vertex[0]),
                    float(vertex[1]),
                    float(vertex[2]),
                )
                if vertex_key not in vertex_indices:
                    vertex_indices[vertex_key] = len(vertices)
                    vertices.append(vertex_key)
                indices.append(vertex_indices[vertex_key])
            triangles.append((indices[0], indices[1], indices[2]))

    with output_path.open("w", encoding="utf-8") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {len(vertices)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write(f"element face {len(triangles)}\n")
        ply_file.write("property list uchar int vertex_indices\n")
        ply_file.write("end_header\n")
        for x_position, y_position, z_position in vertices:
            ply_file.write(f"{x_position} {y_position} {z_position}\n")
        for first_index, second_index, third_index in triangles:
            ply_file.write(f"3 {first_index} {second_index} {third_index}\n")
