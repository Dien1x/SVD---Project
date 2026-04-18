import io
import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk

from core.main_body import load_digit_dataset_from_excel, precompute_digit_svd_bases


file_path = "data.xlsx"
set_type = "zip"

dataset = load_digit_dataset_from_excel(file_path=file_path, set_type=set_type)
svd_cache = precompute_digit_svd_bases(dataset)

current_images = None
current_digit = None
index = 0
svd_preview_cache = {}


root = tk.Tk()
root.title("Digit Viewer")

window_width = 1400
window_height = 750
screen_width = root.winfo_screenwidth()
x = (screen_width // 2) - (window_width // 2)
root.geometry(f"{window_width}x{window_height}+{x}+50")
root.resizable(False, False)

entry = tk.Entry(root, width=5, justify="center", font=("Arial", 14))
entry.pack(pady=5)

digit_label = tk.Label(root, text="", font=("Arial", 20))
digit_label.pack(pady=5)

counter_label = tk.Label(root, text="")
counter_label.pack(pady=5)

img_label = tk.Label(root)
img_label.pack(pady=10)

svd_label = tk.Label(root)
svd_label.pack(pady=10)

show_svd_var = tk.BooleanVar(value=False)


def normalize_to_uint8(img_array):
    img_array = np.asarray(img_array, dtype=float)
    img_min = img_array.min()
    img_max = img_array.max()
    if img_max > img_min:
        img_array = (img_array - img_min) / (img_max - img_min) * 255
    else:
        img_array = np.zeros_like(img_array)
    return img_array.astype(np.uint8)


def build_svd_components_image(digit, img_idx):
    """
    Build an image showing selected rank-1 SVD contributions.
    Cached per (digit, img_idx) for speed.
    """
    cache_key = (digit, img_idx)
    if cache_key in svd_preview_cache:
        return svd_preview_cache[cache_key]

    info = svd_cache[digit]
    U = info["U"]
    S = info["S"]
    Vt = info["V"]

    levels = [1, 2, 4, 8, 16, 32, min(len(S), 64)]
    levels = sorted({k for k in levels if 1 <= k <= len(S)})

    if not levels:
        return None

    fig, axes = plt.subplots(1, len(levels), figsize=(2.0 * len(levels), 2.2))
    if len(levels) == 1:
        axes = [axes]

    for ax, k in zip(axes, levels):
        ui = U[:, k - 1]
        si = S[k - 1]
        vi = Vt[k - 1, img_idx]
        rank1 = (si * ui * vi).reshape(16, 16)
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

    svd_preview_cache[cache_key] = img
    return img


def show_image():
    global index, current_images, current_digit

    if current_images is None:
        return

    img_array = current_images[index].reshape(16, 16)
    img_array = normalize_to_uint8(img_array)

    img = Image.fromarray(img_array).resize((200, 200))
    imgtk = ImageTk.PhotoImage(img)

    img_label.config(image=imgtk)
    img_label.image = imgtk

    digit_label.config(text=f"Digit: {current_digit}")
    counter_label.config(text=f"{index + 1} / {len(current_images)}")

    if show_svd_var.get():
        svd_img = build_svd_components_image(current_digit, index)
        if svd_img is not None:
            svdtk = ImageTk.PhotoImage(svd_img)
            svd_label.config(image=svdtk)
            svd_label.image = svdtk
            return

    svd_label.config(image="")
    svd_label.image = None


def refresh():
    global current_images, current_digit, index

    try:
        digit = int(entry.get())
        if digit not in dataset:
            raise ValueError("Digit must be between 0 and 9.")

        current_digit = digit
        current_images = dataset[digit].T
        index = 0
        show_image()

    except Exception as e:
        digit_label.config(text=str(e))
        counter_label.config(text="")
        img_label.config(image="")
        img_label.image = None
        svd_label.config(image="")
        svd_label.image = None


def next_image():
    global index
    if current_images is not None and index < len(current_images) - 1:
        index += 1
        show_image()


def prev_image():
    global index
    if current_images is not None and index > 0:
        index -= 1
        show_image()


def toggle_svd():
    show_image()


btn_frame = tk.Frame(root)
btn_frame.pack()

prev_btn = tk.Button(btn_frame, text="<-", command=prev_image)
prev_btn.pack(side="left", padx=10)

next_btn = tk.Button(btn_frame, text="->", command=next_image)
next_btn.pack(side="left", padx=10)

refresh_btn = tk.Button(root, text="Refresh", command=refresh)
refresh_btn.pack(pady=10)

svd_checkbox = tk.Checkbutton(
    root,
    text="Show SVD components",
    variable=show_svd_var,
    command=toggle_svd,
)
svd_checkbox.pack(pady=5)

root.mainloop()