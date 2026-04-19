import io
import tkinter as tk
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk

from core.main_body import *

FILE_PATH = "data.xlsx"
SET_TYPE = "zip"
IMG_SIZE = (200, 200)
IMG_SHAPE = (16, 16)
SVD_LEVELS = [1, 2, 4, 8, 16, 32, 64]


class DigitViewer:
    def __init__(self, root: tk.Tk, dataset: dict, svd_cache: dict):
        self.root = root
        self.dataset = dataset
        self.svd_cache = svd_cache

        self.images: np.ndarray | None = None
        self.digit: int | None = None
        self.index: int = 0

        # Precompute valid SVD levels per digit (never changes, so do it once)
        self.svd_levels: dict[int, list[int]] = {
            d: sorted({k for k in SVD_LEVELS if 1 <= k <= len(svd_cache[d]["S"])})
            for d in svd_cache
        }

        self._build_ui()

    # ------------------------------------------------------------------ UI --

    def _build_ui(self):
        root = self.root
        root.title("Digit Viewer")
        root.resizable(False, False)

        w, h = 1400, 750
        x = (root.winfo_screenwidth() - w) // 2
        root.geometry(f"{w}x{h}+{x}+50")

        self.entry = tk.Entry(root, width=5, justify="center", font=("Arial", 14))
        self.entry.pack(pady=5)
        self.entry.bind("<Return>", lambda _: self.refresh())

        self.digit_label = tk.Label(root, text="", font=("Arial", 20))
        self.digit_label.pack(pady=5)

        self.counter_label = tk.Label(root, text="")
        self.counter_label.pack(pady=5)

        self.img_label = tk.Label(root)
        self.img_label.pack(pady=10)

        self.svd_label = tk.Label(root)
        self.svd_label.pack(pady=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack()
        tk.Button(btn_frame, text="<-", command=self.prev_image).pack(side="left", padx=10)
        tk.Button(btn_frame, text="->", command=self.next_image).pack(side="left", padx=10)

        tk.Button(root, text="Refresh", command=self.refresh).pack(pady=10)

        self.show_svd = tk.BooleanVar(value=False)
        tk.Checkbutton(
            root,
            text="Show SVD components",
            variable=self.show_svd,
            command=self._show_image,
        ).pack(pady=5)

    # ------------------------------------------------------------ Handlers --

    def refresh(self):
        try:
            digit = int(self.entry.get())
            if digit not in self.dataset:
                raise ValueError("Digit must be between 0 and 9.")
            self.digit = digit
            self.images = self.dataset[digit].T  # rows = samples
            self.index = 0
            self._show_image()
        except Exception as e:
            self.digit_label.config(text=str(e))
            self.counter_label.config(text="")
            self._clear_image(self.img_label)
            self._clear_image(self.svd_label)

    def next_image(self):
        if self.images is not None and self.index < len(self.images) - 1:
            self.index += 1
            self._show_image()

    def prev_image(self):
        if self.images is not None and self.index > 0:
            self.index -= 1
            self._show_image()

    # --------------------------------------------------------- Rendering ----

    def _show_image(self):
        if self.images is None:
            return

        img_array = self._normalize(self.images[self.index].reshape(IMG_SHAPE))
        photo = self._to_photo(Image.fromarray(img_array).resize(IMG_SIZE))
        self.img_label.config(image=photo)
        self.img_label.image = photo

        self.digit_label.config(text=f"Digit: {self.digit}")
        self.counter_label.config(text=f"{self.index + 1} / {len(self.images)}")

        if self.show_svd.get():
            svd_img = self._build_svd_strip(self.digit, self.index)
            if svd_img is not None:
                photo_svd = self._to_photo(svd_img)
                self.svd_label.config(image=photo_svd)
                self.svd_label.image = photo_svd
                return

        self._clear_image(self.svd_label)

    @lru_cache(maxsize=256)
    def _build_svd_strip(self, digit: int, img_idx: int) -> Image.Image | None:
        info = self.svd_cache[digit]
        U, S, Vt = info["U"], info["S"], info["V"]
        levels = self.svd_levels[digit]
        if not levels:
            return None

        fig, axes = plt.subplots(1, len(levels), figsize=(2.0 * len(levels), 2.2))
        if len(levels) == 1:
            axes = [axes]

        col = Vt[:, img_idx]  # reuse column slice across all k
        for ax, k in zip(axes, levels):
            rank1 = (S[k - 1] * U[:, k - 1] * col[k - 1]).reshape(IMG_SHAPE)
            ax.imshow(rank1, cmap="gray")
            ax.set_title(f"k={k}")
            ax.axis("off")

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img

    # ------------------------------------------------------------ Helpers ---

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
        return np.zeros_like(arr, dtype=np.uint8)

    @staticmethod
    def _to_photo(img: Image.Image) -> ImageTk.PhotoImage:
        return ImageTk.PhotoImage(img)

    @staticmethod
    def _clear_image(label: tk.Label):
        label.config(image="")
        label.image = None


if __name__ == "__main__":
    dataset = load_digit_dataset_from_excel(file_path=FILE_PATH, set_type=SET_TYPE)
    svd_cache = precompute_digit_svd_bases(dataset)

    root = tk.Tk()
    DigitViewer(root, dataset, svd_cache)
    root.mainloop()