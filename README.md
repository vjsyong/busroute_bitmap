# Bus Route Bitmap

Converts Hong Kong bus route sign PNGs (dot-matrix displays) into bitmaps, with the option to re-render them at a custom size and colour.

### Example Output

| Original | Bitmap | Rendered |
|---|---|---|
| ![Original](busroute_bitmap/images/289KY.A2.PNG) | ![Bitmap](images/289KY.A2_bitmap.bmp) | ![Render](images/289KY.A2_bitmap_render.png) |
| Source dot-matrix PNG | 1-pixel-per-dot bitmap (33×128) | Re-rendered at 10px/dot, 2px pitch |

## Setup

```
pip install -r requirements.txt
```

## Usage

### Command line

```python
from main import bus_route_png_to_bmp, render_bitmap

# Convert a sign PNG to a bitmap
bmp_path, bitmap = bus_route_png_to_bmp("input_image/289KY.A2.PNG")

# Re-render with custom settings
render_bitmap(
    bmp_path,
    scalar=10,          # pixels per dot
    dot_pitch=2,        # gap pixels between dots
    on_color=(255, 165, 0),   # RGB for lit dots
    off_color=(32, 16, 0),    # RGB for unlit dots
)
```

### Web UI

```
python app.py
```

Opens a Gradio app where you can:

- Upload a bus sign PNG
- Adjust dot size, spacing, and colours with live preview
- Download the bitmap (.bmp) and rendered image (.png)

## How it works

1. **Crop** — Finds the dense coloured region using row/column projection and crops out the black border.
2. **Detect grid** — Scans for dot runs to determine the grid layout (row/column positions, dot size, pitch) automatically.
3. **Binarize** — Samples the brightness of each dot cell and applies Otsu thresholding to classify dots as ON or OFF.
4. **Render** (optional) — Rebuilds the sign image at a chosen scale, dot spacing, and colour scheme.

## Output files

| File | Description |
|---|---|
| `[name]-bitmap.bmp` | 1-pixel-per-dot binary bitmap (white = ON, black = OFF) |
| `[name]-S[scalar]-P[pitch]-render.png` | Re-rendered sign image |
