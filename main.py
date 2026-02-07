import cv2
import numpy as np
import os


def bus_route_png_to_bmp(input_path: str, output_dir: str = "output_image") -> str:
    """Convert a bus route dot-matrix PNG into a binarized BMP bitmap.

    The function crops tightly into the coloured dot-matrix region, detects the
    dot grid, and binarizes each dot into an ON/OFF pixel using Otsu thresholding.

    Args:
        input_path: Path to the source PNG image.
        output_dir:  Directory to write the output files into.

    Returns:
        Path to the saved .bmp bitmap file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Load ---
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")

    img_h, img_w = image.shape[:2]

    # --- Crop into the dense coloured region ---
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)

    mask = (v > 0).astype(np.uint8)
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)

    row_thresh = img_w * 0.20
    col_thresh = img_h * 0.10

    dense_rows = np.where(row_counts >= row_thresh)[0]
    dense_cols = np.where(col_counts >= col_thresh)[0]

    if len(dense_rows) == 0 or len(dense_cols) == 0:
        raise ValueError("No dense coloured region found in the image.")

    y1, y2 = int(dense_rows[0]), int(dense_rows[-1])
    x1, x2 = int(dense_cols[0]), int(dense_cols[-1])

    margin = 2
    x1 = max(x1 - margin, 0)
    y1 = max(y1 - margin, 0)
    x2 = min(x2 + margin + 1, img_w)
    y2 = min(y2 + margin + 1, img_h)

    cropped = image[y1:y2, x1:x2]

    # --- Detect the dot grid ---
    crop_hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    _, _, crop_v = cv2.split(crop_hsv)
    crop_h, crop_w = crop_v.shape

    # Find a dense data row (one with the most non-zero pixels)
    row_nz_counts = np.count_nonzero(crop_v, axis=1)
    data_row_idx = int(np.argmax(row_nz_counts))
    data_row = crop_v[data_row_idx]

    # Detect column runs (dot positions) from the data row
    nz_cols = np.where(data_row > 0)[0]
    col_starts = [int(nz_cols[0])]
    for i in range(1, len(nz_cols)):
        if nz_cols[i] - nz_cols[i - 1] > 1:
            col_starts.append(int(nz_cols[i]))

    # Measure dot width (median run length) and column pitch
    col_starts_arr = np.array(col_starts)
    col_pitch = int(np.median(np.diff(col_starts_arr))) if len(col_starts_arr) > 1 else 5
    # Dot width = pitch minus the gap (gap = first start of next run - end of previous run)
    run_ends = []
    for cs in col_starts:
        run = np.where(data_row[cs:] == 0)[0]
        run_ends.append(cs + int(run[0]) - 1 if len(run) > 0 else crop_w - 1)
    dot_widths = [e - s + 1 for s, e in zip(col_starts, run_ends)]
    dot_w = int(np.median(dot_widths))

    # Find a dense data column (one with the most non-zero pixels)
    col_nz_counts = np.count_nonzero(crop_v, axis=0)
    data_col_idx = int(np.argmax(col_nz_counts))
    data_col = crop_v[:, data_col_idx]

    # Detect row runs (dot positions) from the data column
    nz_rows = np.where(data_col > 0)[0]
    row_starts = [int(nz_rows[0])]
    for i in range(1, len(nz_rows)):
        if nz_rows[i] - nz_rows[i - 1] > 1:
            row_starts.append(int(nz_rows[i]))

    # Measure dot height (median run length)
    row_run_ends = []
    for rs in row_starts:
        run = np.where(data_col[rs:] == 0)[0]
        row_run_ends.append(rs + int(run[0]) - 1 if len(run) > 0 else crop_h - 1)
    dot_heights = [e - s + 1 for s, e in zip(row_starts, row_run_ends)]
    dot_h = int(np.median(dot_heights))

    n_rows = len(row_starts)
    n_cols = len(col_starts)

    # --- Sample & binarize each dot ---
    dot_values = np.zeros((n_rows, n_cols), dtype=np.uint8)
    for ri, dr in enumerate(row_starts):
        for ci, dc in enumerate(col_starts):
            cell = crop_v[dr : min(dr + dot_h, crop_h), dc : min(dc + dot_w, crop_w)]
            dot_values[ri, ci] = cell.max()

    otsu_thresh, _ = cv2.threshold(
        dot_values.flatten(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    bitmap = (dot_values > otsu_thresh).astype(np.uint8) * 255

    # --- Save outputs ---
    basename = os.path.splitext(os.path.basename(input_path))[0]
    bitmap_path = os.path.join(output_dir, basename + "_bitmap.bmp")
    cropped_path = os.path.join(output_dir, basename + ".png")

    cv2.imwrite(bitmap_path, bitmap)
    cv2.imwrite(cropped_path, cropped)

    print(f"Dot grid : {n_rows} rows × {n_cols} cols = {n_rows * n_cols} dots  (dot size: {dot_w}×{dot_h} px, pitch: {col_pitch} px)")
    print(f"Otsu     : {otsu_thresh:.0f}  →  {(bitmap > 0).sum()} ON / {(bitmap == 0).sum()} OFF")
    print(f"Bitmap   : {bitmap_path}")
    print(f"Cropped  : {cropped_path}")

    return bitmap_path, bitmap


def render_bitmap(
    bitmap_or_path,
    output_path: str = None,
    scalar: int = 10,
    dot_pitch: int = 2,
    on_color: tuple[int, int, int] = (255, 165, 0),
    off_color: tuple[int, int, int] = (32, 16, 0),
) -> np.ndarray:
    """Re-render a dot-matrix bitmap at higher resolution.

    Args:
        bitmap_or_path: Either a file path to a .bmp, or a 2-D NumPy array
                        (grayscale, 0/255) produced by bus_route_png_to_bmp.
        output_path: Where to save the rendered image. ``None`` to skip saving.
        scalar:      Number of pixels per dot (each dot becomes a scalar × scalar square).
        dot_pitch:   Number of blank (off_color) pixels inserted between adjacent dots.
        on_color:    (R, G, B) colour for illuminated dots.
        off_color:   (R, G, B) colour for dark dots.

    Returns:
        The rendered image as a NumPy array (BGR).
    """
    if isinstance(bitmap_or_path, np.ndarray):
        bitmap = bitmap_or_path
    else:
        bitmap = cv2.imread(bitmap_or_path, cv2.IMREAD_GRAYSCALE)
        if bitmap is None:
            raise FileNotFoundError(f"Could not load bitmap: {bitmap_or_path}")

    n_rows, n_cols = bitmap.shape

    # Total size: each dot takes `scalar` px, with `dot_pitch` px gap between dots
    cell = scalar + dot_pitch
    canvas_h = n_rows * cell - dot_pitch  # no trailing gap
    canvas_w = n_cols * cell - dot_pitch

    # Fill canvas with black (gap colour) then paint dots on top
    on_bgr = (on_color[2], on_color[1], on_color[0])
    off_bgr = (off_color[2], off_color[1], off_color[0])
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Paint each dot
    for r in range(n_rows):
        for c in range(n_cols):
            y = r * cell
            x = c * cell
            color = on_bgr if bitmap[r, c] > 0 else off_bgr
            canvas[y : y + scalar, x : x + scalar] = color

    # Save (only when an output path is given or can be derived)
    if output_path is None and isinstance(bitmap_or_path, str):
        base, _ = os.path.splitext(bitmap_or_path)
        output_path = base + "_render.png"
    if output_path is not None:
        cv2.imwrite(output_path, canvas)
        print(f"Rendered : {n_rows}×{n_cols} dots  →  {canvas_w}×{canvas_h} px  (scalar={scalar}, pitch={dot_pitch})")
        print(f"Saved    : {output_path}")

    return canvas


if __name__ == "__main__":
    bmp_path, _ = bus_route_png_to_bmp("input_image/680Y.A4.PNG")
    render_bitmap(bmp_path, scalar=10, dot_pitch=4, on_color=(231, 52, 247), off_color=(33, 8, 36))