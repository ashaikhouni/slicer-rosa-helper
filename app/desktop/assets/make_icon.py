"""Generate the ROSA macOS app icon (build/icon.icns).

A light "ghost" brain profile (facing left) with three cyan SEEG electrodes —
shanks + periodic contact beads — entering from the top, on a dark-slate
rounded-square plate. Palette matches the 3D viewer (dark slate 0x1e2530 +
cyan 0x38d2e6), so the app's face reads like the app itself.

Run on macOS (needs `iconutil`):  python app/desktop/build/make_icon.py
Produces build/icon.icns (wired via electron-builder.yml -> mac.icon) and a
build/icon_preview.png for review. Pure numpy + Pillow; no SVG toolchain.
"""
import os
import subprocess
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops

HERE = os.path.dirname(os.path.abspath(__file__))
S = 1024          # final icon size
SS = 3            # supersample factor for anti-aliasing
W = S * SS

# ---- rounded-square plate with a vertical slate gradient ----
margin = int(W * 0.085)
radius = int(W * 0.225)
plate = (margin, margin, W - margin, W - margin)
PW = plate[2] - plate[0]

top = np.array([42, 54, 70])     # slate, lighter at top
bot = np.array([15, 21, 29])     # darker at bottom
ys = np.linspace(0, 1, W)[:, None, None]
grad = (top[None, None, :] * (1 - ys) + bot[None, None, :] * ys).astype(np.uint8)
grad = np.repeat(grad, W, axis=1)
plate_img = Image.fromarray(grad, "RGB").convert("RGBA")

mask = Image.new("L", (W, W), 0)
ImageDraw.Draw(mask).rounded_rectangle(plate, radius=radius, fill=255)
base = Image.new("RGBA", (W, W), (0, 0, 0, 0))
base.paste(plate_img, (0, 0), mask)


def P(x, y):
    """normalized (0..1 within the plate) -> pixel."""
    return (plate[0] + x * PW, plate[1] + y * PW)


# content transform: enlarge slightly + nudge down so the brain fills the plate
SCALE, DY = 1.10, 0.075
def Q(x, y):
    return P(0.5 + (x - 0.5) * SCALE, 0.5 + (y - 0.5) * SCALE + DY)


def catmull(pts, n=40, closed=False):
    """Sample a Catmull-Rom spline through pts into a dense point list."""
    out = []
    if closed:
        seq = [(pts[(i - 1) % len(pts)], pts[i], pts[(i + 1) % len(pts)],
                pts[(i + 2) % len(pts)]) for i in range(len(pts))]
    else:
        if len(pts) < 3:
            return pts
        ext = [pts[0]] + list(pts) + [pts[-1]]
        seq = [(ext[i - 1], ext[i], ext[i + 1], ext[i + 2]) for i in range(1, len(ext) - 2)]
    for p0, p1, p2, p3 in seq:
        for t in range(n):
            s = t / n; s2, s3 = s * s, s * s * s
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * s + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * s2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * s3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * s + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * s2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * s3)
            out.append((x, y))
    return out


# ---- brain profile (facing left): domed cerebrum, temporal-lobe "thumb",
# Sylvian notch to the frontal lobe, cerebellum lower-right ----
brain_norm = [
    (0.26, 0.24), (0.38, 0.175), (0.52, 0.155), (0.66, 0.185),
    (0.78, 0.27), (0.835, 0.41), (0.79, 0.53), (0.69, 0.595),
    (0.575, 0.615), (0.47, 0.625), (0.35, 0.655), (0.265, 0.585),
    (0.315, 0.505),                                   # Sylvian notch (inward)
    (0.205, 0.455), (0.165, 0.36), (0.205, 0.275),
]
brain_pts = [Q(x, y) for (x, y) in catmull(brain_norm, n=48, closed=True)]

# soft cyan halo behind the brain
halo = Image.new("RGBA", (W, W), (0, 0, 0, 0))
ImageDraw.Draw(halo).polygon(brain_pts, fill=(56, 210, 230, 70))
halo = halo.filter(ImageFilter.GaussianBlur(W * 0.03))
base = Image.alpha_composite(base, halo)

# brain mask (clip sulci strictly inside the silhouette)
brain_mask = Image.new("L", (W, W), 0)
ImageDraw.Draw(brain_mask).polygon(brain_pts, fill=255)

# brain body — light, slightly translucent (ghost)
brain_layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
ImageDraw.Draw(brain_layer).polygon(brain_pts, fill=(214, 223, 232, 235))
base = Image.alpha_composite(base, brain_layer)

# organic gyral bands (no straight grid), clipped to the brain interior
sulci_layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
sd = ImageDraw.Draw(sulci_layer)
sulci = [
    [Q(0.235, 0.40), Q(0.32, 0.435), Q(0.44, 0.415), Q(0.57, 0.44), Q(0.70, 0.42)],  # Sylvian fissure
    [Q(0.27, 0.31), Q(0.39, 0.275), Q(0.50, 0.30), Q(0.62, 0.275), Q(0.71, 0.31)],   # upper gyral band
    [Q(0.31, 0.52), Q(0.42, 0.545), Q(0.55, 0.52), Q(0.66, 0.545)],                  # lower band
    [Q(0.37, 0.345), Q(0.40, 0.39), Q(0.385, 0.43)],                                 # short branch
    [Q(0.585, 0.35), Q(0.61, 0.40)],                                                 # short branch
]
for s in sulci:
    sd.line(catmull(s, n=24), fill=(150, 163, 178, 205), width=int(W * 0.0052), joint="curve")
sulci_layer.putalpha(ImageChops.multiply(sulci_layer.split()[3], brain_mask))
base = Image.alpha_composite(base, sulci_layer)

# ---- electrodes: shanks + periodic contact beads, fanned so strings never overlap ----
CY, BOLT = (56, 210, 230), (150, 240, 250)
elx = Image.new("RGBA", (W, W), (0, 0, 0, 0))
ed = ImageDraw.Draw(elx)
shanks = [
    ((0.40, 0.15), (0.295, 0.505)),   # left,   into temporal/frontal
    ((0.52, 0.135), (0.52, 0.585)),   # centre, deep
    ((0.64, 0.16), (0.725, 0.475)),   # right,  into occipital
]
shaft_w, bead_r = int(W * 0.010), int(W * 0.016)
for en, tn in shanks:
    e = np.array(Q(*en)); t = np.array(Q(*tn))
    ed.line([tuple(e), tuple(t)], fill=CY + (255,), width=shaft_w, joint="curve")
    for f in np.linspace(0.38, 1.0, 6):                     # the SEEG signature
        c = e + (t - e) * f
        ed.ellipse([c[0] - bead_r, c[1] - bead_r, c[0] + bead_r, c[1] + bead_r], fill=CY + (255,))
    br = int(W * 0.025)                                     # entry bolt (brighter)
    ed.ellipse([e[0] - br, e[1] - br, e[0] + br, e[1] + br], fill=BOLT + (255,))
glow = elx.filter(ImageFilter.GaussianBlur(W * 0.012))
base = Image.alpha_composite(base, glow)
base = Image.alpha_composite(base, elx)

# clip everything to the plate so the glow doesn't spill past the rounded corners
clipped = Image.new("RGBA", (W, W), (0, 0, 0, 0))
clipped.paste(base, (0, 0), mask)
icon = clipped.resize((S, S), Image.LANCZOS)
icon.save(os.path.join(HERE, "icon_preview.png"))

# macOS .icns via iconutil (needs the exact iconset filenames)
with tempfile.TemporaryDirectory() as tmp:
    iconset = os.path.join(tmp, "ROSA.iconset")
    os.makedirs(iconset)
    for px, name in [(16, "16x16"), (32, "16x16@2x"), (32, "32x32"), (64, "32x32@2x"),
                     (128, "128x128"), (256, "128x128@2x"), (256, "256x256"), (512, "256x256@2x"),
                     (512, "512x512"), (1024, "512x512@2x")]:
        icon.resize((px, px), Image.LANCZOS).save(os.path.join(iconset, f"icon_{name}.png"))
    icns = os.path.join(HERE, "icon.icns")
    subprocess.run(["iconutil", "-c", "icns", iconset, "-o", icns], check=True)
    print("wrote", icns, "+ icon_preview.png")
