import cv2
import numpy as np
import os


def bus_route_png_to_bmp(input_path: str, output_dir: str = "output_image") -> tuple[str, np.ndarray]:
    """Convert a bus route dot-matrix PNG into a binarized BMP bitmap.

    Works with different sign sizes, aspect ratios, and images that are
    already pre-cropped or have a black border.

    Args:
        input_path: Path to the source PNG image.
        output_dir:  Directory to write the output files into.

    Returns:
        (bitmap_path, bitmap) — path to the saved .bmp and the 2-D NumPy array.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Load ---
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)

    # --- Crop: trim any solid-black border (if present) ---
    # Use a low threshold to include even dim dots
    mask = (v > 5).astype(np.uint8)
    nz_rows = np.where(mask.any(axis=1))[0]
    nz_cols = np.where(mask.any(axis=0))[0]

    if len(nz_rows) == 0 or len(nz_cols) == 0:
        raise ValueError("Image appears entirely black.")

    y1, y2 = int(nz_rows[0]), int(nz_rows[-1]) + 1
    x1, x2 = int(nz_cols[0]), int(nz_cols[-1]) + 1
    cropped = image[y1:y2, x1:x2]

    # --- Threshold the cropped region to isolate dots ---
    crop_hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    _, _, crop_v = cv2.split(crop_hsv)
    crop_h, crop_w = crop_v.shape

    # Otsu to separate bright dots from dark background/gaps
    crop_thresh, crop_bin = cv2.threshold(
        crop_v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --- Detect dot pitch using autocorrelation on projections ---
    # Use binary projection for pitch detection (clean periodic signal)
    bin_col_proj = crop_bin.sum(axis=0).astype(np.float64)
    bin_row_proj = crop_bin.sum(axis=1).astype(np.float64)

    # Use raw V-channel projection for band detection (catches dim dots too)
    raw_col_proj = crop_v.sum(axis=0).astype(np.float64)
    raw_row_proj = crop_v.sum(axis=1).astype(np.float64)

    def _detect_pitch(proj: np.ndarray) -> int:
        """Find the dominant repeating period in a 1-D projection."""
        p = proj - proj.mean()
        corr = np.correlate(p, p, mode="full")
        corr = corr[len(corr) // 2:]  # keep positive lags
        # Zero out lag 0, find first peak after that
        corr[0] = 0
        # Minimum pitch is 3 pixels (smallest reasonable dot+gap)
        min_lag = 3
        if len(corr) <= min_lag:
            return len(proj)
        peak = int(np.argmax(corr[min_lag:])) + min_lag
        return peak

    col_pitch = _detect_pitch(bin_col_proj)
    row_pitch = _detect_pitch(bin_row_proj)

    # --- Build the grid by finding dot-band centroids ---
    def _find_dot_bands(proj_1d: np.ndarray, pitch: int) -> list[int]:
        """Find dot centre positions by locating bright bands in the projection.

        1. Threshold the projection to find bright bands (contiguous runs).
        2. Filter out bands whose width is far from the median (noise).
        3. Compute the centroid of each surviving band.
        4. Keep only bands that fit a regular grid with the detected pitch.
        5. Fill in all positions at pitch intervals between the first and last
           band so that columns/rows with only dim (below-Otsu) dots are
           not lost.
        """
        # Adaptive threshold with baseline subtraction: handles non-black
        # backgrounds by removing the uniform base contribution.
        baseline = float(np.median(np.sort(proj_1d)[:max(1, len(proj_1d) // 4)]))
        adjusted = proj_1d - baseline
        thresh = np.clip(adjusted, 0, None).max() * 0.10
        bright = adjusted > thresh

        # Find contiguous bright runs (bands)
        bands: list[tuple[int, int]] = []  # (start, end) inclusive
        in_band = False
        start = 0
        for i in range(len(bright)):
            if bright[i] and not in_band:
                start = i
                in_band = True
            elif not bright[i] and in_band:
                bands.append((start, i - 1))
                in_band = False
        if in_band:
            bands.append((start, len(bright) - 1))

        if not bands:
            return []

        # Filter bands by width: remove noise bands whose width is very
        # different from the median.  Real dots have consistent widths.
        widths = np.array([e - s + 1 for s, e in bands])
        med_w = float(np.median(widths))
        bands = [b for b, w in zip(bands, widths) if 0.4 * med_w <= w <= 2.5 * med_w]
        if not bands:
            return []

        # Centroid of each band (weighted by projection value)
        centroids = []
        for s, e in bands:
            seg = proj_1d[s : e + 1]
            total = seg.sum()
            if total > 0:
                centroids.append(s + float(np.sum(np.arange(len(seg)) * seg) / total))
            else:
                centroids.append((s + e) / 2.0)

        # Find the longest subsequence of centroids with ~regular spacing (pitch)
        centroids_arr = np.array(centroids)
        if len(centroids_arr) < 2:
            return [int(round(centroids_arr[0]))]

        gaps = np.diff(centroids_arr)
        # A gap is "on-grid" if it's close to a multiple of pitch
        tolerance = pitch * 0.35
        on_grid = np.array([
            abs(g - round(g / pitch) * pitch) < tolerance and round(g / pitch) >= 1
            for g in gaps
        ])

        # Find the longest consecutive run of on-grid gaps
        best_start, best_len = 0, 0
        cur_start, cur_len = 0, 0
        for i, ok in enumerate(on_grid):
            if ok:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_start = cur_start
                    best_len = cur_len
            else:
                cur_len = 0

        # Use the first and last band centroids as grid boundaries, then fill
        # at regular pitch intervals so dim (below-Otsu) positions are kept.
        first_c = centroids_arr[best_start]
        last_c = centroids_arr[best_start + best_len]
        n_dots = int(round((last_c - first_c) / pitch)) + 1
        starts = [int(round(first_c + i * pitch)) for i in range(n_dots)]
        return starts

    col_starts = _find_dot_bands(raw_col_proj, col_pitch)
    row_starts = _find_dot_bands(raw_row_proj, row_pitch)

    if not col_starts or not row_starts:
        raise ValueError("Could not detect a regular dot grid in the image.")

    # Measure actual dot width / height from binary image at those positions
    def _measure_dot_size(crop_bin: np.ndarray, starts: list[int], axis: int, fallback: int) -> int:
        """Measure the median bright-run length at grid positions."""
        sizes = []
        for s in starts:
            if axis == 1:  # measuring width along columns
                line = crop_bin[:, min(s, crop_w - 1)]
            else:  # measuring height along rows
                line = crop_bin[min(s, crop_h - 1), :]
            bright_px = np.where(line > 0)[0]
            if len(bright_px) > 0:
                runs = np.split(bright_px, np.where(np.diff(bright_px) > 1)[0] + 1)
                sizes.append(int(np.median([len(r) for r in runs])))
        return int(np.median(sizes)) if sizes else fallback

    dot_w = min(_measure_dot_size(crop_bin, col_starts, 1, col_pitch), col_pitch)
    dot_h = min(_measure_dot_size(crop_bin, row_starts, 0, row_pitch), row_pitch)

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

    print(f"Dot grid : {n_rows} rows x {n_cols} cols = {n_rows * n_cols} dots  (dot: {dot_w}x{dot_h} px, pitch: {col_pitch}x{row_pitch} px)")
    print(f"Otsu     : {otsu_thresh:.0f}  ->  {(bitmap > 0).sum()} ON / {(bitmap == 0).sum()} OFF")
    print(f"Bitmap   : {bitmap_path}")
    print(f"Cropped  : {cropped_path}")

    return bitmap_path, bitmap


def render_bitmap(
    bitmap_or_path,
    output_path: str = None,
    scalar: int = 10,
    dot_pitch: int = 2,
    on_color: tuple[int, int, int] = (255, 165, 0),
    off_color: tuple[int, int, int] = (24, 12, 0),
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
        print(f"Rendered : {n_rows}x{n_cols} dots  ->  {canvas_w}x{canvas_h} px  (scalar={scalar}, pitch={dot_pitch})")
        print(f"Saved    : {output_path}")

    return canvas


if __name__ == "__main__":
    bmp_path, _ = bus_route_png_to_bmp("input_image/681-S1.PNG")
    render_bitmap(bmp_path, scalar=10, dot_pitch=4, on_color=(231, 52, 247), off_color=(33, 8, 36))