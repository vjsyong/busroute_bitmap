import gradio as gr
import numpy as np
import cv2
import tempfile
import os

import re

from main import bus_route_png_to_bmp, render_bitmap

# Module-level cache so we only re-process when a new image is uploaded
_cache: dict = {"path": None, "bitmap": None}


def _parse_color(color_str) -> tuple[int, int, int]:
    """Parse a colour value from Gradio into an (R, G, B) tuple."""
    if color_str is None:
        return (255, 255, 255)

    s = str(color_str).strip()

    # Hex: "#RRGGBB" or "#RGB"
    if s.startswith("#"):
        h = s.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) >= 6:
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    # CSS: "rgb(R, G, B)" or "rgba(R, G, B, A)" — values may be floats
    m = re.match(
        r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)", s
    )
    if m:
        return (
            min(int(round(float(m.group(1)))), 255),
            min(int(round(float(m.group(2)))), 255),
            min(int(round(float(m.group(3)))), 255),
        )

    # Fallback
    return (255, 255, 255)


def _base_name(file_path: str) -> str:
    """Extract the base name (without extension) from an upload path."""
    return os.path.splitext(os.path.basename(file_path))[0]


def _process_upload(file_path: str) -> np.ndarray | None:
    """Run the PNG → bitmap pipeline, caching the result."""
    if file_path is None:
        _cache["path"] = None
        _cache["bitmap"] = None
        return None

    if file_path != _cache["path"]:
        _, bitmap = bus_route_png_to_bmp(file_path, output_dir=tempfile.mkdtemp())
        _cache["path"] = file_path
        _cache["bitmap"] = bitmap

    return _cache["bitmap"]


def _get_bitmap_file(file_path: str) -> str | None:
    """Return a path to the cached bitmap as a downloadable .bmp file."""
    bitmap = _process_upload(file_path)
    if bitmap is None:
        return None

    name = _base_name(file_path)
    tmp_path = os.path.join(tempfile.gettempdir(), f"{name}-bitmap.bmp")
    cv2.imwrite(tmp_path, bitmap)
    return tmp_path


def _render(
    file_path: str,
    scalar: int,
    dot_pitch: int,
    on_color: str,
    off_color: str,
) -> tuple[np.ndarray | None, str | None]:
    """Render the cached bitmap with the current settings."""
    bitmap = _process_upload(file_path)
    if bitmap is None:
        return None, None

    sc = int(scalar)
    dp = int(dot_pitch)

    # Parse colour string → (R, G, B)
    on_rgb = _parse_color(on_color)
    off_rgb = _parse_color(off_color)

    canvas_bgr = render_bitmap(
        bitmap,
        output_path=None,
        scalar=sc,
        dot_pitch=dp,
        on_color=on_rgb,
        off_color=off_rgb,
    )

    # Save render to a temp file with the required naming
    name = _base_name(file_path)
    render_path = os.path.join(
        tempfile.gettempdir(), f"{name}-S{sc}-P{dp}-render.png"
    )
    cv2.imwrite(render_path, canvas_bgr)

    # Convert BGR → RGB for Gradio display
    return cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB), render_path


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Bus Route Bitmap Renderer") as app:
        gr.Markdown("## 🚌 Bus Route Dot-Matrix → Bitmap Renderer")
        gr.Markdown(
            "Upload a bus route dot-matrix PNG, then tweak the render settings. "
            "The preview updates in real time."
        )

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.Image(
                    label="Upload PNG",
                    type="filepath",
                    sources=["upload"],
                )
                scalar = gr.Slider(
                    minimum=1,
                    maximum=30,
                    value=10,
                    step=1,
                    label="Scalar (px per dot)",
                )
                dot_pitch = gr.Slider(
                    minimum=0,
                    maximum=10,
                    value=2,
                    step=1,
                    label="Dot pitch (gap px)",
                )
                on_color = gr.ColorPicker(
                    label="ON dot colour",
                    value="#FFA500",
                )
                off_color = gr.ColorPicker(
                    label="OFF dot colour",
                    value="#201000",
                )

            with gr.Column(scale=3):
                preview = gr.Image(label="Rendered preview", type="numpy")
                with gr.Row():
                    bitmap_download = gr.File(label="Download bitmap (.bmp)", interactive=False)
                    render_download = gr.File(label="Download render (.png)", interactive=False)

        # All inputs that should trigger a live re-render
        inputs = [file_input, scalar, dot_pitch, on_color, off_color]
        render_outputs = [preview, render_download]

        # Wire every control to the render function
        for ctrl in inputs:
            ctrl.change(fn=_render, inputs=inputs, outputs=render_outputs)

        # Provide the bitmap file for download whenever a new image is uploaded
        file_input.change(fn=_get_bitmap_file, inputs=[file_input], outputs=bitmap_download)

    return app


if __name__ == "__main__":
    build_app().launch()
