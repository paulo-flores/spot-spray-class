import io
import csv
import hashlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from streamlit_image_coordinates import streamlit_image_coordinates


# ==================== CLASS IMAGE SETTINGS ====================
# Put ONE image in each folder:
#   images/very-low/
#   images/low/
#   images/medium/
#   images/high/
IMAGE_ROOT = Path(__file__).parent / "images"
LEVEL_DIRS = {
    "Very-low": "very-low",
    "Low": "low",
    "Medium": "medium",
    "High": "high",
}
VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def find_single_image_for_level(level_label: str) -> Optional[Path]:
    folder = IMAGE_ROOT / LEVEL_DIRS[level_label]
    if not folder.exists():
        return None
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    files.sort()
    if not files:
        return None
    return files[0]  # deterministic if more than one exists


def load_repo_image(level_label: str) -> Tuple[Optional[Image.Image], Optional[str], Optional[str]]:
    p = find_single_image_for_level(level_label)
    if p is None:
        return None, None, None
    img = Image.open(p).convert("RGB")
    # hash bytes to detect changes across reruns (fast enough for classroom images)
    h = hashlib.md5(img.tobytes()).hexdigest()
    return img, p.name, h


# ---------------- helpers ----------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normalize_rect(p1, p2):
    x0, y0 = p1
    x1, y1 = p2
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def compute_grid_dims(aoi_w_ft, aoi_h_ft, cell_w_ft, cell_h_ft):
    nx = max(1, int(np.ceil(aoi_w_ft / cell_w_ft)))
    ny = max(1, int(np.ceil(aoi_h_ft / cell_h_ft)))
    return nx, ny


def click_to_cell(x, y, aoi_px, nx, ny):
    x0, y0, x1, y1 = aoi_px
    if not (x0 <= x <= x1 and y0 <= y <= y1):
        return None
    w = x1 - x0
    h = y1 - y0
    if w <= 0 or h <= 0:
        return None
    cw = w / nx
    ch = h / ny
    c = int((x - x0) / cw)
    r = int((y - y0) / ch)
    c = clamp(c, 0, nx - 1)
    r = clamp(r, 0, ny - 1)
    return (r, c)


def resample_selection(old, new_ny, new_nx):
    old_ny, old_nx = old.shape
    out = np.zeros((new_ny, new_nx), dtype=bool)
    for r in range(new_ny):
        rr = int(round((r + 0.5) * old_ny / new_ny - 0.5))
        rr = clamp(rr, 0, old_ny - 1)
        for c in range(new_nx):
            cc = int(round((c + 0.5) * old_nx / new_nx - 0.5))
            cc = clamp(cc, 0, old_nx - 1)
            out[r, c] = old[rr, cc]
    return out


def selection_csv_bytes(selected):
    output = io.StringIO()
    w = csv.writer(output)
    w.writerow(["row", "col", "spray_cell (1=yes,0=no)"])
    ny, nx = selected.shape
    for r in range(ny):
        for c in range(nx):
            w.writerow([r, c, int(selected[r, c])])
    return output.getvalue().encode("utf-8")


def crop_view(img, cx, cy, view_w, view_h):
    """Crop a viewport centered at (cx,cy) in full-image pixels."""
    W, H = img.size
    x0 = int(round(cx - view_w / 2))
    y0 = int(round(cy - view_h / 2))
    x0 = clamp(x0, 0, W - view_w)
    y0 = clamp(y0, 0, H - view_h)
    x1 = x0 + view_w
    y1 = y0 + view_h
    return img.crop((x0, y0, x1, y1)), x0, y0  # also return offsets


def draw_overlay_on_view(
    view_img,
    view_offset,
    aoi_px,
    nx,
    ny,
    selected,
    show_grid=True,
    first_corner=None,
    grid_outer=7,
    grid_inner=5,
    aoi_outer=8,
    aoi_inner=5,
    sel_outer=12,
    sel_inner=7,
):
    """
    Draw AOI + grid + selection outlines onto a VIEWPORT image.
    Coordinates are mapped from full-image -> viewport via view_offset.
    """
    img = view_img.copy()
    d = ImageDraw.Draw(img)
    ox, oy = view_offset

    # first corner marker (full-image coords) -> view coords
    if first_corner is not None and aoi_px is None:
        fx, fy = first_corner
        vx, vy = fx - ox, fy - oy
        r = 12
        d.ellipse([vx - r, vy - r, vx + r, vy + r], outline=(0, 0, 0), width=7)
        d.ellipse([vx - r, vy - r, vx + r, vy + r], outline=(255, 255, 0), width=4)

    if aoi_px is None:
        return img

    # AOI rect in view coords
    x0, y0, x1, y1 = aoi_px
    x0v, y0v, x1v, y1v = x0 - ox, y0 - oy, x1 - ox, y1 - oy

    # If AOI is completely outside view, nothing to draw
    if x1v < 0 or y1v < 0 or x0v > img.size[0] or y0v > img.size[1]:
        return img

    # AOI outline
    d.rectangle([x0v, y0v, x1v, y1v], outline=(0, 0, 0), width=aoi_outer)
    d.rectangle([x0v, y0v, x1v, y1v], outline=(0, 255, 255), width=aoi_inner)

    w = x1 - x0
    h = y1 - y0
    if w <= 0 or h <= 0:
        return img

    cw = w / nx
    ch = h / ny

    if show_grid:
        for i in range(1, nx):
            x = int(x0 + i * cw) - ox
            d.line([x, y0v, x, y1v], fill=(0, 0, 0), width=grid_outer)
            d.line([x, y0v, x, y1v], fill=(255, 255, 0), width=grid_inner)
        for j in range(1, ny):
            y = int(y0 + j * ch) - oy
            d.line([x0v, y, x1v, y], fill=(0, 0, 0), width=grid_outer)
            d.line([x0v, y, x1v, y], fill=(255, 255, 0), width=grid_inner)

    # FAST: draw only selected cells
    if selected is not None:
        ys, xs = np.where(selected)
        for r, c in zip(ys.tolist(), xs.tolist()):
            rx0 = int(x0 + c * cw) - ox
            ry0 = int(y0 + r * ch) - oy
            rx1 = int(x0 + (c + 1) * cw) - ox
            ry1 = int(y0 + (r + 1) * ch) - oy
            d.rectangle([rx0, ry0, rx1, ry1], outline=(0, 0, 0), width=sel_outer)
            d.rectangle([rx0, ry0, rx1, ry1], outline=(255, 0, 0), width=sel_inner)

    return img


# ---------------- app ----------------
st.set_page_config(page_title="Spot Spray Activity", layout="wide")
st.title("Spot-Spray Classroom Activity")

ss = st.session_state
ss.setdefault("img", None)
ss.setdefault("img_name", None)
ss.setdefault("img_hash", None)

ss.setdefault("aoi_clicks", [])
ss.setdefault("aoi_px", None)  # full-image coords
ss.setdefault("selected", None)  # bool [ny,nx]

ss.setdefault("status", "")
ss.setdefault("last_raw_click", None)

# Viewport state (pan/zoom)
ss.setdefault("view_zoom", 1.0)  # 1.0 = full view; larger = zoom in
ss.setdefault("view_cx", None)
ss.setdefault("view_cy", None)

# Level state
ss.setdefault("level", "Low")


with st.sidebar:
    st.header("Inputs")

    st.subheader("Class image")
    level = st.radio("Weed infestation level", ["Very-low", "Low", "Medium", "High"], index=1)
    ss["level"] = level

    aoi_w_ft = st.number_input("AOI width (ft)", value=100.0, min_value=0.1, step=1.0)
    aoi_h_ft = st.number_input("AOI height (ft)", value=50.0, min_value=0.1, step=1.0)

    cell_w_ft = st.number_input("Cell width (ft)", value=5.0, min_value=0.1, step=0.5)
    cell_h_ft = st.number_input("Cell height (ft)", value=5.0, min_value=0.1, step=0.5)

    acres = st.number_input("Field acres", value=640.0, min_value=0.0, step=10.0)
    cost_per_ac = st.number_input("Herbicide cost ($/acre)", value=15.0, min_value=0.0, step=1.0)

    st.divider()
    mode = st.radio("Mode", ["Set AOI (2 clicks)", "Mark weeds (toggle cells)"], index=0)

    # Viewport controls
    st.subheader("Zoom / Pan")
    zoom = st.slider("Zoom", 1.0, 8.0, float(ss["view_zoom"]), 0.25)
    ss["view_zoom"] = zoom

    pan_step = st.slider("Pan step (px)", 50, 800, 200, 50)

    colp1, colp2, colp3 = st.columns(3)
    if colp2.button("⬆️"):
        ss["view_cy"] = None if ss["view_cy"] is None else ss["view_cy"] - pan_step
    colp4, colp5, colp6 = st.columns(3)
    if colp4.button("⬅️"):
        ss["view_cx"] = None if ss["view_cx"] is None else ss["view_cx"] - pan_step
    if colp6.button("➡️"):
        ss["view_cx"] = None if ss["view_cx"] is None else ss["view_cx"] + pan_step
    colp7, colp8, colp9 = st.columns(3)
    if colp8.button("⬇️"):
        ss["view_cy"] = None if ss["view_cy"] is None else ss["view_cy"] + pan_step

    st.caption("Tip: click a point in the image, then use arrows to pan around that area.")

    st.divider()
    show_grid = st.checkbox("Show grid lines", value=True)
    debug = st.checkbox("Debug (show click payload)", value=False)

    st.divider()
    if st.button("Reset AOI"):
        ss["aoi_clicks"] = []
        ss["aoi_px"] = None
        ss["selected"] = None
        ss["status"] = "AOI reset."
        ss["last_raw_click"] = None

    if st.button("Clear selections"):
        if ss["selected"] is not None:
            ss["selected"][:, :] = False
        ss["status"] = "Selections cleared."
        ss["last_raw_click"] = None


# ---------------- Load image from repo based on selected level ----------------
img, name, img_hash = load_repo_image(ss["level"])
if img is None:
    st.error(
        f"No image found for level '{ss['level']}'. "
        f"Expected 1 image in: images/{LEVEL_DIRS[ss['level']]}/"
    )
    st.stop()

# Only reset if the image actually changed
if ss["img_hash"] != img_hash:
    ss["img"] = img
    ss["img_name"] = name
    ss["img_hash"] = img_hash

    W, H = img.size
    ss["view_cx"] = W // 2
    ss["view_cy"] = H // 2
    ss["view_zoom"] = 1.0

    ss["aoi_clicks"] = []
    ss["aoi_px"] = None
    ss["selected"] = None
    ss["status"] = f"Loaded: {ss['level']} ({name}). Set AOI with two clicks."
    ss["last_raw_click"] = None

st.caption(f"Loaded: **{ss['level']}** → `{ss['img_name']}`")


# ---------------- Main interaction ----------------
base_img = ss["img"]
W, H = base_img.size

if ss["view_cx"] is None:
    ss["view_cx"] = W // 2
if ss["view_cy"] is None:
    ss["view_cy"] = H // 2

# Clamp viewport center to image
ss["view_cx"] = clamp(int(ss["view_cx"]), 0, W - 1)
ss["view_cy"] = clamp(int(ss["view_cy"]), 0, H - 1)

# Compute grid dims (can change live without wiping selections)
nx, ny = compute_grid_dims(aoi_w_ft, aoi_h_ft, cell_w_ft, cell_h_ft)
if ss["aoi_px"] is not None:
    if ss["selected"] is None:
        ss["selected"] = np.zeros((ny, nx), dtype=bool)
    elif ss["selected"].shape != (ny, nx):
        ss["selected"] = resample_selection(ss["selected"], ny, nx)

# Bigger image (Option A style; results below)
disp_w = 2200
disp_w = clamp(disp_w, 1100, 2600)
disp_h = int(disp_w * (H / W))

view_w = int(W / ss["view_zoom"])
view_h = int(H / ss["view_zoom"])
view_w = clamp(view_w, 200, W)
view_h = clamp(view_h, 200, H)

view_img, ox, oy = crop_view(base_img, ss["view_cx"], ss["view_cy"], view_w, view_h)

# Scale factors from displayed viewport -> full-image
sx = view_w / float(disp_w)
sy = view_h / float(disp_h)

first_corner = ss["aoi_clicks"][0] if len(ss["aoi_clicks"]) == 1 else None

overlay_view = draw_overlay_on_view(
    view_img,
    (ox, oy),
    ss["aoi_px"],
    nx,
    ny,
    ss["selected"],
    show_grid=show_grid,
    first_corner=first_corner,
).resize((disp_w, disp_h))

st.subheader("Interact")
if mode == "Set AOI (2 clicks)":
    st.info("Click corner 1 and corner 2 of the AOI (inside the viewport).")
else:
    st.info("Click cells to toggle spray (red outline).")

clicked = streamlit_image_coordinates(overlay_view, width=disp_w, key="img_clicks")

if debug:
    st.write("clicked:", clicked)
    st.write("aoi_clicks:", ss["aoi_clicks"])
    st.write("aoi_px:", ss["aoi_px"])
    st.write("view offset:", (ox, oy))
    st.write("view zoom:", ss["view_zoom"])

# Handle click exactly once per new click
if isinstance(clicked, dict) and "x" in clicked and "y" in clicked:
    raw = (int(clicked["x"]), int(clicked["y"]))
    if raw != ss["last_raw_click"]:
        ss["last_raw_click"] = raw

        # Map click from displayed viewport -> full image coords
        x_full = int(ox + clicked["x"] * sx)
        y_full = int(oy + clicked["y"] * sy)
        x_full = clamp(x_full, 0, W - 1)
        y_full = clamp(y_full, 0, H - 1)

        # set pan center to last click
        ss["view_cx"] = x_full
        ss["view_cy"] = y_full

        if mode == "Set AOI (2 clicks)":
            # redefine AOI if already set and user starts over
            if ss["aoi_px"] is not None and len(ss["aoi_clicks"]) == 0:
                ss["aoi_px"] = None
                ss["selected"] = None

            if len(ss["aoi_clicks"]) == 0:
                ss["aoi_clicks"] = [(x_full, y_full)]
                ss["status"] = "Corner 1 set. Now click the opposite corner."
            else:
                p1 = ss["aoi_clicks"][0]
                p2 = (x_full, y_full)
                ss["aoi_px"] = normalize_rect(p1, p2)
                ss["aoi_clicks"] = []
                ss["selected"] = np.zeros((ny, nx), dtype=bool)
                ss["status"] = "AOI set. Switch to weeds mode."
        else:
            if ss["aoi_px"] is None:
                ss["status"] = "Set AOI first (switch to AOI mode)."
            else:
                cell = click_to_cell(x_full, y_full, ss["aoi_px"], nx, ny)
                if cell is None:
                    ss["status"] = "Clicked outside AOI."
                else:
                    r, c = cell
                    ss["selected"][r, c] = ~ss["selected"][r, c]
                    ss["status"] = f"Toggled cell r={r}, c={c}."

        # immediate redraw
        st.rerun()

if ss.get("status"):
    st.success(ss["status"])


# ---------------- Results BELOW the image (Option A) ----------------
st.markdown("---")
st.subheader("Results")

if ss["aoi_px"] is None or ss["selected"] is None:
    st.warning("Set the AOI to see calculations.")
    st.stop()

total = nx * ny
sprayed = int(ss["selected"].sum())
unsprayed = total - sprayed
savings_frac = (unsprayed / total) if total else 0.0
savings_pct = savings_frac * 100.0

full_cost = acres * cost_per_ac
field_savings = savings_frac * full_cost

st.markdown(
    f"""
**Infestation level:** {ss["level"]}  
**Image:** {ss["img_name"]}  
**AOI (ft):** {aoi_w_ft:g} W × {aoi_h_ft:g} H  
**Cell (ft):** {cell_w_ft:g} W × {cell_h_ft:g} H  
**Grid:** {nx} × {ny} (cells: {total})  
**Spray cells:** {sprayed}  
**No-spray cells:** {unsprayed}  
**Savings:** {savings_pct:0.1f}%  

**Field acres:** {acres:g}  
**$/acre:** ${cost_per_ac:g}  
**Full-field herbicide cost:** ${full_cost:,.2f}  
✅ **Potential savings:** **${field_savings:,.2f}**
"""
)

st.download_button(
    "Download CSV",
    data=selection_csv_bytes(ss["selected"]),
    file_name="spot_spray_grid.csv",
    mime="text/csv",
)
