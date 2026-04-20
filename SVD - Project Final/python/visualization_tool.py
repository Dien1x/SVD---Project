# ===== Εργαλείο οπτικοποίησης ψηφίων με SVD ανάλυση =====
# ===== Φόρτωση dataset από Excel, εμφάνιση εικόνων και SVD components =====
# ===== Αλληλεπιδραστικό GUI με Tkinter για πλοήγηση στα δεδομένα =====


import io
import tkinter as tk
from functools import lru_cache

import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image, ImageTk

from main_body import *

import ctypes
import platform

# ----------------------------------------------------------------------------------
# DPI AWARENESS — Windows only
# ----------------------------------------------------------------------------------
# Στα Windows με high-DPI οθόνες (π.χ. 125%, 150% scaling), το Tkinter εμφανίζει
# θολό κείμενο και θολά widgets γιατί τα Windows κάνουν bitmap upscaling της
# εφαρμογής αντί να την αφήσουν να σχεδιάσει στη native ανάλυση.
# Η παρακάτω κλήση δηλώνει στα Windows ότι η εφαρμογή είναι "DPI-aware",
# οπότε το Tkinter λαμβάνει τις πραγματικές διαστάσεις οθόνης και σχεδιάζει
# με crisp κείμενο και widgets.
if platform.system() == "Windows":
    try:
        # SetProcessDpiAwareness(2) = PROCESS_PER_MONITOR_DPI_AWARE (Windows 8.1+)
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            # Fallback για παλαιότερα Windows: SetProcessDPIAware
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


# ----------------------------------------------------------------------------------
# ΓΕΝΙΚΗ ΙΔΕΑ ΚΑΙ ΣΤΡΑΤΗΓΙΚΗ
# ----------------------------------------------------------------------------------
# Αυτό το αρχείο υλοποιεί ένα αλληλεπιδραστικό GUI για την οπτικοποίηση ψηφίων
# από ένα dataset που φορτώνεται από αρχείο Excel. Ο χρήστης μπορεί να:
#
#   — Επιλέξει ένα ψηφίο (0–9) και να πλοηγηθεί στα δείγματά του.
#   — Δει κάθε δείγμα ως εικόνα 16×16 εικονοστοιχείων (pixels), μεγεθυμένη σε 200×200.
#   — Ενεργοποιήσει την εμφάνιση των SVD components για το τρέχον δείγμα,
#     δηλαδή τις rank-1 συνεισφορές για διάφορες τιμές k.
#
# Η κλάση DigitViewer οργανώνει όλη τη λογική και την κατάσταση της εφαρμογής,
# αποφεύγοντας καθολικές μεταβλητές. Η κατασκευή του GUI γίνεται στη μέθοδο
# _build_ui(), ενώ η λογική εμφάνισης και πλοήγησης βρίσκεται στις υπόλοιπες μεθόδους.
#
# Σημαντική βελτιστοποίηση: τα SVD strip images αποθηκεύονται σε cache με lru_cache,
# ώστε να μη χρειάζεται να ξαναυπολογίζονται όταν ο χρήστης επιστρέφει σε δείγμα
# που έχει ήδη δει.
# ----------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------
# ΣΤΑΘΕΡΕΣ ΕΦΑΡΜΟΓΗΣ
# ----------------------------------------------------------------------------------

# Διαδρομή του αρχείου Excel που περιέχει το dataset των ψηφίων
script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(script_dir, "..", "data.xlsx")

# Τύπος συμπίεσης του dataset μέσα στο Excel (χρησιμοποιείται από τη load_digit_dataset_from_excel)
SET_TYPE = "zip"

# Μέγεθος εμφάνισης κάθε εικόνας ψηφίου στο GUI (πλάτος × ύψος σε pixels)
IMG_SIZE = (200, 200)

# Πραγματικές διαστάσεις κάθε δείγματος: κάθε δείγμα είναι διάνυσμα 256 στοιχείων
# που αναδιαμορφώνεται σε τετραγωνικό πλέγμα 16×16
IMG_SHAPE = (16, 16)

# Τιμές k για τις οποίες εμφανίζουμε τη rank-1 SVD συνεισφορά στο SVD strip.
# Επιλέχθηκαν ώστε να καλύπτουν λογαριθμικά το εύρος των δυνατών k,
# δείχνοντας πώς αλλάζει η αναπαράσταση καθώς προσθέτουμε περισσότερα components.
SVD_LEVELS = [1, 2, 4, 8, 16, 32, 64]


# ----------------------------------------------------------------------------------
# ΚΛΑΣΗ DigitViewer
# ----------------------------------------------------------------------------------

class DigitViewer:
    """
    Κύρια κλάση της εφαρμογής οπτικοποίησης ψηφίων.

    Διαχειρίζεται το GUI (Tkinter), την πλοήγηση στα δείγματα του dataset,
    και την εμφάνιση των SVD components για κάθε δείγμα.

    Attributes
    ----------
    root : tk.Tk
        Το κύριο παράθυρο της εφαρμογής.
    dataset : dict
        digit -> np.ndarray σχήματος (256, n_samples), τα δεδομένα των ψηφίων.
    svd_cache : dict
        digit -> {"U": U, "S": S, "V": Vt, "max_k": max_k}, τα αποτελέσματα της SVD.
    images : np.ndarray | None
        Τα δείγματα του τρέχοντος ψηφίου ως πίνακας (n_samples, 256).
        None αν δεν έχει φορτωθεί ακόμα ψηφίο.
    digit : int | None
        Το τρέχον επιλεγμένο ψηφίο (0–9). None αρχικά.
    index : int
        Ο δείκτης του τρέχοντος δείγματος μέσα στο images.
    svd_levels : dict
        digit -> list[int], οι έγκυρες τιμές k για κάθε ψηφίο (υποσύνολο του SVD_LEVELS).
    """

    def __init__(self, root: tk.Tk, dataset: dict, svd_cache: dict):
        """
        Αρχικοποίηση της εφαρμογής.

        Αποθηκεύει τα δεδομένα, προϋπολογίζει τα έγκυρα SVD levels για κάθε ψηφίο,
        και κατασκευάζει το GUI.

        Parameters
        ----------
        root : tk.Tk
            Το κύριο παράθυρο Tkinter.
        dataset : dict
            digit -> np.ndarray (256, n_samples), τα δεδομένα των ψηφίων.
        svd_cache : dict
            digit -> {"U": U, "S": S, "V": Vt, "max_k": max_k}.
        """
        self.root = root
        self.dataset = dataset
        self.svd_cache = svd_cache

        # Αρχικοποίηση κατάστασης πλοήγησης: κανένα ψηφίο δεν έχει επιλεγεί ακόμα
        self.images: np.ndarray | None = None
        self.digit: int | None = None
        self.index: int = 0

        # Προϋπολογισμός των έγκυρων SVD levels για κάθε ψηφίο.
        # Για κάθε ψηφίο, κρατάμε μόνο τις τιμές k από το SVD_LEVELS που δεν ξεπερνούν
        # τον αριθμό των ιδιαζουσών τιμών που υπάρχουν στο svd_cache για αυτό το ψηφίο.
        # Γίνεται μία φορά εδώ και όχι σε κάθε κλήση του _build_svd_strip, για να αποφύγουμε
        # επαναλαμβανόμενο υπολογισμό.
        self.svd_levels: dict[int, list[int]] = {
            d: sorted({k for k in SVD_LEVELS if 1 <= k <= len(svd_cache[d]["S"])})
            for d in svd_cache
        }

        # Κατασκευή του GUI
        self._build_ui()

    # ------------------------------------------------------------------
    # ΚΑΤΑΣΚΕΥΗ GUI
    # ------------------------------------------------------------------

    def _build_ui(self):
        """
        Κατασκευή όλων των widgets του GUI και ορισμός διαστάσεων παραθύρου.

        Η διάταξη των widgets από πάνω προς τα κάτω είναι:
            1. Entry        — πεδίο εισαγωγής ψηφίου (0–9)
            2. Label        — εμφάνιση τρέχοντος ψηφίου
            3. Label        — μετρητής δείγματος (π.χ. "3 / 50")
            4. Label        — εικόνα τρέχοντος δείγματος
            5. Label        — εικόνα SVD strip (ορατή μόνο όταν είναι ενεργό το checkbox)
            6. Frame        — κουμπιά πλοήγησης <- και ->
            7. Button       — Refresh (φόρτωση νέου ψηφίου)
            8. Checkbutton  — ενεργοποίηση/απενεργοποίηση SVD components
        """
        root = self.root
        root.title("Digit Viewer")

        # Απενεργοποίηση αλλαγής μεγέθους παραθύρου για σταθερή διάταξη
        root.resizable(False, False)

        # Ορισμός σταθερών διαστάσεων παραθύρου και οριζόντια κεντράρισμα στην οθόνη
        w, h = 1400, 750
        x = (root.winfo_screenwidth() - w) // 2
        root.geometry(f"{w}x{h}+{x}+50")

        # Πεδίο εισαγωγής ψηφίου — ο χρήστης πληκτρολογεί έναν αριθμό 0–9.
        # Το <Return> δεσμεύεται στη μέθοδο refresh() ώστε να μη χρειάζεται κλικ στο κουμπί
        self.entry = tk.Entry(root, width=5, justify="center", font=("Arial", 14))
        self.entry.pack(pady=5)
        self.entry.bind("<Return>", lambda _: self.refresh())

        # Label εμφάνισης τρέχοντος ψηφίου (π.χ. "Digit: 3")
        self.digit_label = tk.Label(root, text="", font=("Arial", 20))
        self.digit_label.pack(pady=5)

        # Label μετρητή δείγματος (π.χ. "5 / 50")
        self.counter_label = tk.Label(root, text="")
        self.counter_label.pack(pady=5)

        # Label εμφάνισης της κανονικοποιημένης και μεγεθυμένης εικόνας του δείγματος
        self.img_label = tk.Label(root)
        self.img_label.pack(pady=10)

        # Label εμφάνισης του SVD strip — ορατό μόνο όταν είναι ενεργό το checkbox
        self.svd_label = tk.Label(root)
        self.svd_label.pack(pady=10)

        # Κουμπιά πλοήγησης: <- για προηγούμενο δείγμα, -> για επόμενο
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        tk.Button(btn_frame, text="<-", command=self.prev_image).pack(side="left", padx=10)
        tk.Button(btn_frame, text="->", command=self.next_image).pack(side="left", padx=10)

        # Κουμπί Refresh: φορτώνει τα δείγματα για το ψηφίο που έχει πληκτρολογηθεί
        tk.Button(root, text="Refresh", command=self.refresh).pack(pady=10)

        # Checkbox για ενεργοποίηση/απενεργοποίηση εμφάνισης SVD components.
        # Η τιμή αποθηκεύεται στο BooleanVar show_svd και διαβάζεται στο _show_image()
        self.show_svd = tk.BooleanVar(value=False)
        tk.Checkbutton(
            root,
            text="Show SVD components",
            variable=self.show_svd,
            command=self._show_image,   # ενημέρωση GUI αμέσως στην αλλαγή του checkbox
        ).pack(pady=5)

    # ------------------------------------------------------------------
    # HANDLERS ΠΛΟΗΓΗΣΗΣ
    # ------------------------------------------------------------------

    def refresh(self):
        """
        Φόρτωση των δειγμάτων για το ψηφίο που έχει πληκτρολογηθεί στο entry.

        Διαβάζει την τιμή του entry, ελέγχει αν είναι έγκυρο ψηφίο (0–9),
        φορτώνει τα αντίστοιχα δείγματα από το dataset, μηδενίζει τον δείκτη
        και εμφανίζει το πρώτο δείγμα.

        Σε περίπτωση σφάλματος (μη αριθμός ή ψηφίο εκτός εύρους), εμφανίζει
        το μήνυμα σφάλματος στο digit_label και καθαρίζει τα image labels.
        """
        try:
            digit = int(self.entry.get())

            # Έλεγχος αν το ψηφίο υπάρχει στο dataset
            if digit not in self.dataset:
                raise ValueError("Digit must be between 0 and 9.")

            self.digit = digit

            # Φόρτωση δειγμάτων: το dataset αποθηκεύει τα δείγματα ως στήλες (256, n_samples),
            # οπότε κάνουμε transpose για να έχουμε κάθε γραμμή ως ένα δείγμα (n_samples, 256)
            self.images = self.dataset[digit].T

            # Επαναφορά δείκτη στο πρώτο δείγμα
            self.index = 0

            self._show_image()

        except Exception as e:
            # Εμφάνιση σφάλματος στο GUI και καθαρισμός εικόνων
            self.digit_label.config(text=str(e))
            self.counter_label.config(text="")
            self._clear_image(self.img_label)
            self._clear_image(self.svd_label)

    def next_image(self):
        """
        Μετάβαση στο επόμενο δείγμα του τρέχοντος ψηφίου.

        Αυξάνει τον δείκτη κατά 1, αν δεν έχουμε φτάσει στο τέλος των δειγμάτων.
        Δεν κάνει τίποτα αν δεν έχει φορτωθεί ψηφίο ή αν είμαστε στο τελευταίο δείγμα.
        """
        if self.images is not None and self.index < len(self.images) - 1:
            self.index += 1
            self._show_image()

    def prev_image(self):
        """
        Μετάβαση στο προηγούμενο δείγμα του τρέχοντος ψηφίου.

        Μειώνει τον δείκτη κατά 1, αν δεν έχουμε φτάσει στην αρχή των δειγμάτων.
        Δεν κάνει τίποτα αν δεν έχει φορτωθεί ψηφίο ή αν είμαστε στο πρώτο δείγμα.
        """
        if self.images is not None and self.index > 0:
            self.index -= 1
            self._show_image()

    # ------------------------------------------------------------------
    # ΕΜΦΑΝΙΣΗ ΕΙΚΟΝΩΝ
    # ------------------------------------------------------------------

    def _show_image(self):
        """
        Ενημέρωση του GUI με την εικόνα και τα SVD components του τρέχοντος δείγματος.

        Εκτελείται κάθε φορά που αλλάζει το τρέχον δείγμα (πλοήγηση, refresh)
        ή η κατάσταση του checkbox SVD.

        Λογική:
            1. Αναδιαμορφώνει το διάνυσμα 256 στοιχείων σε πίνακα 16×16.
            2. Κανονικοποιεί σε [0, 255] και εμφανίζει ως εικόνα 200×200.
            3. Ενημερώνει digit_label και counter_label.
            4. Αν το SVD checkbox είναι ενεργό, εμφανίζει το SVD strip.
               Αλλιώς, καθαρίζει το svd_label.
        """
        # Αν δεν έχει φορτωθεί ακόμα ψηφίο, δεν κάνουμε τίποτα
        if self.images is None:
            return

        # Αναδιαμόρφωση τρέχοντος δείγματος από (256,) σε (16, 16) και κανονικοποίηση
        img_array = self._normalize(self.images[self.index].reshape(IMG_SHAPE))

        # Μετατροπή σε PIL Image, μεγέθυνση σε 200×200 και μετατροπή σε PhotoImage για το Tkinter
        photo = self._to_photo(Image.fromarray(img_array).resize(IMG_SIZE))

        # Ενημέρωση του label εικόνας — αποθηκεύουμε αναφορά στο photo για να μην
        # σβηστεί από τον garbage collector του Python (γνωστό πρόβλημα με Tkinter)
        self.img_label.config(image=photo)
        self.img_label.image = photo

        # Ενημέρωση label ψηφίου και μετρητή
        self.digit_label.config(text=f"Digit: {self.digit}")
        self.counter_label.config(text=f"{self.index + 1} / {len(self.images)}")

        # Αν το checkbox είναι ενεργό, εμφανίζουμε το SVD strip για το τρέχον δείγμα
        if self.show_svd.get():
            svd_img = self._build_svd_strip(self.digit, self.index)
            if svd_img is not None:
                photo_svd = self._to_photo(svd_img)
                self.svd_label.config(image=photo_svd)
                self.svd_label.image = photo_svd
                return

        # Αν το checkbox είναι ανενεργό (ή το svd_img ήταν None), καθαρίζουμε το svd_label
        self._clear_image(self.svd_label)

    @lru_cache(maxsize=256)
    def _build_svd_strip(self, digit: int, img_idx: int) -> Image.Image | None:
        """
        Κατασκευή εικόνας που δείχνει τις rank-1 SVD συνεισφορές για διάφορα k.

        Για κάθε τιμή k στο self.svd_levels[digit], υπολογίζουμε τη rank-1 συνεισφορά
        του k-οστού SVD component στο τρέχον δείγμα:

            rank1 = S[k-1] * U[:, k-1] * Vt[k-1, img_idx]

        όπου:
            U[:, k-1]        — ο k-οστός αριστερός ιδιάζων διάνυσμα (βάση του υποχώρου)
            S[k-1]           — η k-οστή ιδιάζουσα τιμή (σπουδαιότητα του component)
            Vt[k-1, img_idx] — ο συντελεστής προβολής του συγκεκριμένου δείγματος
                               στον k-οστό ιδιάζοντα διάνυσμα

        Το αποτέλεσμα αναδιαμορφώνεται σε 16×16 και εμφανίζεται ως grayscale subplot.

        Η μέθοδος αποθηκεύεται σε cache μέσω @lru_cache(maxsize=256), οπότε για το ίδιο
        (digit, img_idx) ζεύγος δεν ξαναυπολογίζεται αν ο χρήστης επιστρέψει σε αυτό.

        Parameters
        ----------
        digit : int
            Το τρέχον ψηφίο (0–9).
        img_idx : int
            Ο δείκτης του τρέχοντος δείγματος μέσα στο self.images.

        Returns
        -------
        Image.Image | None
            PIL Image με το SVD strip, ή None αν δεν υπάρχουν έγκυρα levels για αυτό το ψηφίο.
        """
        info = self.svd_cache[digit]
        U, S, Vt = info["U"], info["S"], info["V"]

        # Παίρνουμε τα προϋπολογισμένα έγκυρα levels για αυτό το ψηφίο
        levels = self.svd_levels[digit]
        if not levels:
            return None

        # Δημιουργία figure με έναν subplot για κάθε τιμή k
        fig, axes = plt.subplots(1, len(levels), figsize=(2.0 * len(levels), 2.2))

        # Αν υπάρχει μόνο ένα level, το subplots επιστρέφει ένα μοναδικό Axes αντικείμενο,
        # το οποίο τυλίγουμε σε λίστα για ομοιόμορφη επεξεργασία στο loop
        if len(levels) == 1:
            axes = [axes]

        # Εξαγωγή της στήλης img_idx από τον πίνακα Vt μία φορά, εκτός του loop.
        # Αυτή η στήλη περιέχει τους συντελεστές προβολής του συγκεκριμένου δείγματος
        # σε όλους τους ιδιάζοντες διάνυσμα — διαβάζουμε col[k-1] για κάθε k στο loop
        col = Vt[:, img_idx]

        # Σχεδίαση κάθε rank-1 component
        for ax, k in zip(axes, levels):
            # Υπολογισμός rank-1 συνεισφοράς: πολλαπλασιασμός ιδιάζοντος διανύσματος (U[:, k-1]),
            # ιδιάζουσας τιμής (S[k-1]) και συντελεστή προβολής (col[k-1]) του δείγματος.
            # Το αποτέλεσμα είναι ένα διάνυσμα 256 στοιχείων που αναδιαμορφώνεται σε 16×16
            rank1 = (S[k - 1] * U[:, k - 1] * col[k - 1]).reshape(IMG_SHAPE)
            ax.imshow(rank1, cmap="gray")
            ax.set_title(f"k={k}")
            ax.axis("off")

        plt.tight_layout()

        # Αποθήκευση του figure σε buffer μνήμης (PNG), κλείσιμο του figure για αποδέσμευση
        # μνήμης, και άνοιγμα ως PIL Image. Το .copy() αποσυνδέει την εικόνα από τον buffer
        # πριν τον κλείσουμε, ώστε να μη χαθούν τα δεδομένα
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()

        return img

    # ------------------------------------------------------------------
    # ΒΟΗΘΗΤΙΚΕΣ ΜΕΘΟΔΟΙ
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """
        Κανονικοποίηση πίνακα τιμών στο εύρος [0, 255] (uint8).

        Εφαρμόζει min-max scaling: η ελάχιστη τιμή γίνεται 0 και η μέγιστη 255.
        Αν όλες οι τιμές είναι ίδιες (επίπεδη εικόνα), επιστρέφεται μηδενικός πίνακας.

        Parameters
        ----------
        arr : np.ndarray
            Πίνακας αριθμητικών τιμών οποιουδήποτε εύρους.

        Returns
        -------
        np.ndarray
            Πίνακας uint8 με τιμές στο [0, 255].
        """
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)

        # Αν η εικόνα είναι επίπεδη (ίδια τιμή παντού), επιστρέφουμε μαύρη εικόνα
        return np.zeros_like(arr, dtype=np.uint8)

    @staticmethod
    def _to_photo(img: Image.Image) -> ImageTk.PhotoImage:
        """
        Μετατροπή PIL Image σε Tkinter PhotoImage.

        Χρησιμοποιείται για τη φόρτωση εικόνων στα Label widgets του GUI.

        Parameters
        ----------
        img : Image.Image
            Η εικόνα προς μετατροπή.

        Returns
        -------
        ImageTk.PhotoImage
            Αντικείμενο εικόνας συμβατό με Tkinter.
        """
        return ImageTk.PhotoImage(img)

    @staticmethod
    def _clear_image(label: tk.Label):
        """
        Καθαρισμός εικόνας από ένα Tkinter Label.

        Αφαιρεί τόσο το config image όσο και την αποθηκευμένη αναφορά (.image),
        ώστε να αποδεσμευτεί η μνήμη από το PhotoImage αντικείμενο.

        Parameters
        ----------
        label : tk.Label
            Το Label widget που θέλουμε να καθαρίσουμε.
        """
        label.config(image="")
        label.image = None


# ----------------------------------------------------------------------------------
# ΣΗΜΕΙΟ ΕΙΣΟΔΟΥ
# ----------------------------------------------------------------------------------

if __name__ == "__main__":
    # Φόρτωση dataset από το Excel αρχείο
    dataset = load_digit_dataset_from_excel(file_path=FILE_PATH, set_type=SET_TYPE)

    # Προϋπολογισμός SVD για όλα τα ψηφία (0–9) — γίνεται μία φορά πριν το GUI
    svd_cache = precompute_digit_svd_bases(dataset)

    # Δημιουργία κύριου παραθύρου Tkinter και εκκίνηση εφαρμογής
    root = tk.Tk()
    DigitViewer(root, dataset, svd_cache)
    root.mainloop()