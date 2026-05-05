"""Split combined Figure 1.1 into (a) device chain + partition, (b) pipelined schedule."""
from pathlib import Path

from PIL import Image
import numpy as np

SRC = Path(
    r"C:\Users\runcong\.cursor\projects\c-Users-runcong-Desktop\assets"
    r"\c__Users_runcong_AppData_Roaming_Cursor_User_workspaceStorage_7647fa47f058504c06793151efe8bcd9_images_image-82cf94e3-6c6a-403e-89e0-471bf089947c.png"
)
OUT_DIR = Path(__file__).resolve().parent.parent / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def last_content_row(rgb: np.ndarray, end_y: int, thresh: float = 0.012) -> int:
    """Last row index in [0, end_y) with significant non-white ink."""
    gray = rgb.mean(axis=2)
    for y in range(end_y - 1, -1, -1):
        if (gray[y, :] < 245).mean() > thresh:
            return y
    return 0


def main() -> None:
    im = Image.open(SRC).convert("RGBA")
    w, h = im.size
    rgb = np.array(im.convert("RGB"))

    # Title for pipelined panel begins ~y=328; cut just above to keep title wholly in (b)
    y_split = 324

    top = im.crop((0, 0, w, y_split))
    bottom = im.crop((0, y_split, w, h))

    top_rgb = np.array(top.convert("RGB"))
    y_last = last_content_row(top_rgb, y_split)
    pad = 14
    top_trim_h = min(y_last + pad, y_split)
    top = top.crop((0, 0, w, top_trim_h))

    out_a = OUT_DIR / "figure_1_1a_device_chain_and_partition.png"
    out_b = OUT_DIR / "figure_1_1b_pipelined_execution.png"
    top.save(out_a)
    bottom.save(out_b)
    print("Saved:", out_a)
    print("Saved:", out_b)
    print(
        "source",
        w,
        h,
        "split_y:",
        y_split,
        "trimmed top height:",
        top_trim_h,
        "bottom height:",
        bottom.height,
    )


if __name__ == "__main__":
    main()
