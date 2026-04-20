# ===== Εργαλείο ζωγραφικής και αναγνώρισης ψηφίων με SVD =====
# ===== Φόρτωση digit_u_dict από .pkl, σχεδίαση ψηφίου και πρόβλεψη =====
# ===== Αλληλεπιδραστικό GUI με Tkinter — χωρίς εξάρτηση από dataset =====


import ctypes
import pickle
import platform
import tkinter as tk
from tkinter import filedialog

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageTk

from evaluation import predict_digit_by_relative_residual


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
# Αυτό το αρχείο υλοποιεί ένα αυτόνομο GUI για την αναγνώριση χειρόγραφων ψηφίων.
# Ο χρήστης δεν χρειάζεται πρόσβαση στο αρχικό dataset — αρκεί το αρχείο
# digit_u_dict.pkl που παράγεται από προηγούμενα στάδια εκπαίδευσης.
#
# Ροή εργασίας:
#
#   Βήμα 1 — Φόρτωση μοντέλου:
#       Ο χρήστης επιλέγει ένα .pkl αρχείο μέσω file dialog. Το αρχείο περιέχει
#       ένα digit_u_dict: dict[int, {"k": int, "U_cols": np.ndarray}], δηλαδή
#       τις πρώτες k στήλες του U για κάθε ψηφίο από την SVD ανάλυση.
#
#   Βήμα 2 — Σχεδίαση:
#       Ο χρήστης ζωγραφίζει ένα ψηφίο στον καμβά (canvas) με το ποντίκι.
#       Χρησιμοποιούμε ένα PIL Image στο παρασκήνιο για να αποθηκεύουμε
#       ακριβώς την ίδια εικόνα που σχεδιάζεται στον καμβά του Tkinter.
#
#   Βήμα 3 — Προεπεξεργασία:
#       Όταν ο χρήστης πατήσει "Predict", η εικόνα του καμβά μετατρέπεται σε
#       grayscale, κλιμακώνεται σε 16×16 (με αντιολίσθηση LANCZOS), και
#       κανονικοποιείται σε διάνυσμα 256 στοιχείων με τιμές στο [0, 1].
#
#   Βήμα 4 — Πρόβλεψη:
#       Το διάνυσμα περνά στη συνάρτηση predict_digit_by_relative_residual(),
#       η οποία υπολογίζει το σχετικό υπόλειμμα προβολής για κάθε ψηφίο και
#       επιστρέφει το ψηφίο με το μικρότερο υπόλειμμα (= καλύτερη αντιστοίχιση).
#
# Σημείωση σχεδιασμού:
#       Ο καμβάς του Tkinter δεν αποθηκεύει εικόνα — χρησιμοποιούμε παράλληλο
#       PIL ImageDraw για pixel-accurate αποθήκευση της ζωγραφιάς, ανεξάρτητα
#       από τα anti-aliasing του οθόνη.
# ----------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------
# ΣΤΑΘΕΡΕΣ ΕΦΑΡΜΟΓΗΣ
# ----------------------------------------------------------------------------------

# Διαστάσεις καμβά στον οποίο ζωγραφίζει ο χρήστης (σε pixels)
# Μεγαλύτερος καμβάς = πιο άνετη σχεδίαση, κλιμακώνεται σε 16×16 κατά την πρόβλεψη
CANVAS_SIZE = 280

# Μέγεθος της εικόνας στην οποία κλιμακώνεται ο καμβάς για την πρόβλεψη (16×16 = 256 pixels)
MODEL_IMG_SIZE = (16, 16)

# Μέγεθος της εικόνας preview (μεγεθυμένη έκδοση του 16×16 για εμφάνιση στο GUI)
PREVIEW_SIZE = (112, 112)

# Πάχος πινέλου σχεδίασης στον καμβά (σε pixels)
BRUSH_RADIUS = 8

# Χρώμα φόντου καμβά (μαύρο, όπως τα MNIST δεδομένα)
BG_COLOR = "black"

# Χρώμα πινέλου (λευκό πάνω σε μαύρο φόντο, όπως τα MNIST δεδομένα)
FG_COLOR = "white"


# ----------------------------------------------------------------------------------
# ΚΛΑΣΗ DigitPredictor
# ----------------------------------------------------------------------------------

class DigitPredictor:
    """
    Κύρια κλάση της εφαρμογής αναγνώρισης χειρόγραφων ψηφίων.

    Διαχειρίζεται το GUI (Tkinter), τη σχεδίαση στον καμβά, την προεπεξεργασία
    της εικόνας και την κλήση του μοντέλου πρόβλεψης.

    Attributes
    ----------
    root : tk.Tk
        Το κύριο παράθυρο της εφαρμογής.
    digit_u_dict : dict | None
        Το φορτωμένο μοντέλο: digit -> {"k": int, "U_cols": np.ndarray}.
        None αν δεν έχει φορτωθεί ακόμα αρχείο.
    pil_image : PIL.Image.Image
        Παράλληλη PIL εικόνα στο παρασκήνιο, στην οποία αποθηκεύεται ακριβώς
        ό,τι σχεδιάζεται στον καμβά. Χρησιμοποιείται για την εξαγωγή του
        διανύσματος 256 στοιχείων κατά την πρόβλεψη.
    draw : PIL.ImageDraw.Draw
        Αντικείμενο σχεδίασης για την pil_image.
    last_x, last_y : int | None
        Τελευταία γνωστή θέση ποντικιού κατά τη σχεδίαση, για τη σύνδεση
        διαδοχικών σημείων σε συνεχή γραμμή.
    """

    def __init__(self, root: tk.Tk):
        """
        Αρχικοποίηση της εφαρμογής.

        Δημιουργεί την κενή κατάσταση, αρχικοποιεί την PIL εικόνα παρασκηνίου
        και κατασκευάζει το GUI.

        Parameters
        ----------
        root : tk.Tk
            Το κύριο παράθυρο Tkinter.
        """
        self.root = root

        # Το μοντέλο είναι None μέχρι ο χρήστης να φορτώσει αρχείο .pkl
        self.digit_u_dict: dict | None = None

        # Αρχικοποίηση PIL εικόνας παρασκηνίου: μαύρο φόντο, ίδιο μέγεθος με τον καμβά
        # Αυτή είναι η "αληθινή" εικόνα που θα επεξεργαστούμε — ο καμβάς Tkinter
        # είναι μόνο για την οπτική εμφάνιση στον χρήστη
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.pil_image)

        # Τελευταία θέση ποντικιού — χρησιμοποιείται για τη σχεδίαση συνεχών γραμμών
        self.last_x: int | None = None
        self.last_y: int | None = None

        # Στοίβα αναίρεσης: αποθηκεύει αντίγραφα της PIL εικόνας πριν από κάθε stroke
        self.undo_stack: list[Image.Image] = []

        self._build_ui()

    # ------------------------------------------------------------------
    # ΚΑΤΑΣΚΕΥΗ GUI
    # ------------------------------------------------------------------

    def _build_ui(self):
        """
        Κατασκευή όλων των widgets του GUI.

        Διάταξη (αριστερά | δεξιά):
            Αριστερή στήλη:
                — Label οδηγιών φόρτωσης
                — Button φόρτωσης .pkl
                — Label ονόματος φορτωμένου αρχείου
                — Canvas σχεδίασης (CANVAS_SIZE × CANVAS_SIZE)
                — Frame κουμπιών (Predict | Clear)

            Δεξιά στήλη:
                — Label "Preview (16×16)"
                — Label εικόνας preview (μεγεθυμένο 16×16)
                — Label "Prediction"
                — Label αποτελέσματος πρόβλεψης (μεγάλος αριθμός)
                — Label λεπτομερειών (residuals για κάθε ψηφίο)
        """
        root = self.root
        root.title("Digit Predictor")
        root.resizable(False, False)

        # Κεντράρισμα παραθύρου οριζόντια στην οθόνη
        # Αυξημένο ύψος ώστε τα κουμπιά να μην κόβονται σε high-DPI οθόνες
        w, h = 820, 530
        x = (root.winfo_screenwidth() - w) // 2
        root.geometry(f"{w}x{h}+{x}+80")

        # ── Αριστερό πλαίσιο: φόρτωση αρχείου + καμβάς σχεδίασης ──────────
        left_frame = tk.Frame(root, padx=15, pady=10)
        left_frame.pack(side="left", fill="y")

        tk.Label(left_frame, text="1.  Φόρτωσε το μοντέλο (.pkl)", font=("Arial", 11)).pack(anchor="w")

        # Κουμπί φόρτωσης .pkl — ανοίγει file dialog για επιλογή αρχείου
        tk.Button(
            left_frame,
            text="📂  Load digit_u_dict.pkl",
            command=self._load_model,
            width=24,
            font=("Arial", 10),
        ).pack(pady=(2, 0))

        # Label που εμφανίζει το όνομα του φορτωμένου αρχείου (ή "Κανένα αρχείο")
        self.file_label = tk.Label(
            left_frame, text="Κανένα αρχείο φορτωμένο", fg="gray", font=("Arial", 9)
        )
        self.file_label.pack(pady=(0, 8))

        tk.Label(left_frame, text="2.  Ζωγράφισε ένα ψηφίο (0–9)", font=("Arial", 11)).pack(anchor="w")

        # Καμβάς Tkinter για τη σχεδίαση — μαύρο φόντο, σταθερό μέγεθος
        self.canvas = tk.Canvas(
            left_frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg=BG_COLOR,
            cursor="crosshair",
        )
        self.canvas.pack(pady=(2, 6))

        # Δέσμευση συμβάντων ποντικιού για τη σχεδίαση
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_press)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)

        # Πλήκτρα πρόσβασης: Enter → Predict, Ctrl+Z → Undo
        root.bind("<Return>", lambda e: self._predict())
        root.bind("<Control-z>", self._undo)

        # Κουμπιά "Predict" και "Clear" δίπλα-δίπλα, με αρκετό padding ώστε
        # να μην κόβονται σε high-DPI οθόνες
        btn_frame = tk.Frame(left_frame)
        btn_frame.pack(pady=(6, 2))

        tk.Button(
            btn_frame,
            text="🔍  Predict",
            command=self._predict,
            width=13,
            font=("Arial", 11, "bold"),
            pady=4,
        ).pack(side="left", padx=8)

        tk.Button(
            btn_frame,
            text="🗑  Clear",
            command=self._clear_canvas,
            width=13,
            font=("Arial", 11),
            pady=4,
        ).pack(side="left", padx=8)

        # ── Δεξί πλαίσιο: preview εικόνας + αποτέλεσμα πρόβλεψης ──────────
        right_frame = tk.Frame(root, padx=20, pady=10)
        right_frame.pack(side="left", fill="both", expand=True)

        tk.Label(right_frame, text="Preview  16 × 16", font=("Arial", 11, "bold")).pack(pady=(10, 2))

        # Label εμφάνισης του μεγεθυμένου 16×16 — δείχνει στον χρήστη
        # ακριβώς τι "βλέπει" το μοντέλο μετά την κλιμάκωση
        self.preview_label = tk.Label(right_frame, bg="#222222", width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1])
        self.preview_label.pack(pady=(0, 12))

        tk.Label(right_frame, text="Πρόβλεψη", font=("Arial", 11, "bold")).pack()

        # Μεγάλο label για την κύρια πρόβλεψη (εμφανίζει τον αριθμό 0–9)
        self.result_label = tk.Label(
            right_frame, text="—", font=("Arial", 64, "bold"), fg="#1a73e8"
        )
        self.result_label.pack()

        # Label λεπτομερειών: εμφανίζει το residual κάθε ψηφίου για debugging
        self.detail_label = tk.Label(
            right_frame, text="", font=("Arial", 9), fg="#555555", justify="left"
        )
        self.detail_label.pack(pady=(4, 0))

    # ------------------------------------------------------------------
    # ΦΟΡΤΩΣΗ ΜΟΝΤΕΛΟΥ
    # ------------------------------------------------------------------

    def _load_model(self):
        """
        Άνοιγμα file dialog για επιλογή .pkl αρχείου και φόρτωση του digit_u_dict.

        Το .pkl αρχείο πρέπει να περιέχει ένα dict της μορφής:
            digit (int) -> {"k": int, "U_cols": np.ndarray σχήματος (256, k)}

        Σε περίπτωση σφάλματος (λάθος μορφή, κατεστραμμένο αρχείο κ.λπ.),
        εμφανίζει το μήνυμα σφάλματος στο file_label.
        """
        # Άνοιγμα file dialog — φιλτράρουμε μόνο .pkl αρχεία για ευκολία
        path = filedialog.askopenfilename(
            title="Επιλογή digit_u_dict.pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
        )

        # Αν ο χρήστης ακύρωσε το dialog, δεν κάνουμε τίποτα
        if not path:
            return

        try:
            with open(path, "rb") as f:
                self.digit_u_dict = pickle.load(f)

            # Εμφάνιση μόνο του ονόματος αρχείου (όχι ολόκληρης της διαδρομής)
            filename = path.split("/")[-1]
            self.file_label.config(
                text=f"✅  {filename}", fg="green"
            )
            # Καθαρισμός τυχόν προηγούμενου αποτελέσματος
            self.result_label.config(text="—")
            self.detail_label.config(text="")

        except Exception as e:
            self.digit_u_dict = None
            self.file_label.config(text=f"❌  Σφάλμα: {e}", fg="red")

    # ------------------------------------------------------------------
    # ΣΧΕΔΙΑΣΗ ΣΤΟΝ ΚΑΜΒΑ
    # ------------------------------------------------------------------

    def _on_mouse_press(self, event):
        """
        Καταγραφή αρχικής θέσης ποντικιού όταν πατηθεί το αριστερό κουμπί.

        Σχεδιάζει μια μικρή κουκκίδα στο σημείο κλικ, ώστε το πάτημα χωρίς
        drag να αφήνει κι αυτό ορατό σημάδι.

        Parameters
        ----------
        event : tk.Event
            Συμβάν ποντικιού με πεδία .x και .y (θέση σε pixels στον καμβά).
        """
        self.last_x, self.last_y = event.x, event.y

        # Αποθήκευση snapshot πριν από κάθε νέο stroke (για Ctrl+Z αναίρεση)
        self.undo_stack.append(self.pil_image.copy())

        # Σχεδίαση κουκκίδας στο σημείο κλικ (για απλό tap χωρίς drag)
        self._draw_point(event.x, event.y)

    def _on_mouse_drag(self, event):
        """
        Σχεδίαση γραμμής από την τελευταία γνωστή θέση ποντικιού ως την τρέχουσα.

        Κατά το drag, συνδέουμε διαδοχικά σημεία με γεμιστά ελλείψεις (κύκλους)
        κατά μήκος της γραμμής, δημιουργώντας ομαλή συνεχή γραμμή.

        Parameters
        ----------
        event : tk.Event
            Συμβάν ποντικιού με πεδία .x και .y.
        """
        if self.last_x is None:
            return

        # Σχεδίαση γεμιστής γραμμής στον Tkinter καμβά (για οπτική εμφάνιση)
        self.canvas.create_line(
            self.last_x, self.last_y, event.x, event.y,
            fill=FG_COLOR,
            width=BRUSH_RADIUS * 2,
            capstyle=tk.ROUND,
            smooth=True,
        )

        # Παράλληλη σχεδίαση στην PIL εικόνα παρασκηνίου (για αποθήκευση pixel τιμών)
        self.draw.line(
            [self.last_x, self.last_y, event.x, event.y],
            fill=255,
            width=BRUSH_RADIUS * 2,
        )

        self.last_x, self.last_y = event.x, event.y

    def _on_mouse_release(self, event):
        """
        Επαναφορά τελευταίας θέσης ποντικιού μετά την αποδέσμευση του κουμπιού.

        Μηδενίζει last_x/last_y ώστε η επόμενη πίεση να ξεκινά νέα γραμμή
        αντί να συνδεθεί με την προηγούμενη.

        Parameters
        ----------
        event : tk.Event
            Συμβάν ποντικιού (δεν χρησιμοποιείται, αλλά απαιτείται από το Tkinter).
        """
        self.last_x = None
        self.last_y = None
        # Ενημέρωση preview αμέσως μετά το stroke — ο χρήστης βλέπει σε
        # πραγματικό χρόνο την κεντραρισμένη εικόνα που θα δει το μοντέλο
        self._update_live_preview()

    def _draw_point(self, x: int, y: int):
        """
        Σχεδίαση μιας κουκκίδας (κύκλου) στο σημείο (x, y).

        Χρησιμοποιείται για την αρχή κάθε stroke (mouse press χωρίς drag),
        ώστε ένα απλό κλικ να αφήνει ορατό σημάδι.

        Parameters
        ----------
        x : int
            Οριζόντια συντεταγμένη σε pixels στον καμβά.
        y : int
            Κατακόρυφη συντεταγμένη σε pixels στον καμβά.
        """
        r = BRUSH_RADIUS

        # Σχεδίαση στον Tkinter καμβά
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=FG_COLOR, outline=FG_COLOR)

        # Σχεδίαση στην PIL εικόνα παρασκηνίου
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def _clear_canvas(self):
        """
        Καθαρισμός καμβά και επαναφορά σε κενή κατάσταση.

        Σβήνει όλα τα σχεδιαστικά στοιχεία από τον Tkinter καμβά, δημιουργεί
        νέα μαύρη PIL εικόνα παρασκηνίου, και καθαρίζει τα labels αποτελέσματος.
        """
        # Αποθήκευση snapshot ώστε το Clear να μπορεί να αναιρεθεί με Ctrl+Z
        self.undo_stack.append(self.pil_image.copy())

        # Διαγραφή όλων των στοιχείων από τον Tkinter καμβά
        self.canvas.delete("all")

        # Δημιουργία νέας κενής PIL εικόνας παρασκηνίου
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.pil_image)

        # Καθαρισμός αποτελέσματος και preview
        self.result_label.config(text="—")
        self.detail_label.config(text="")
        self.preview_label.config(image="", width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1])
        self.preview_label.image = None

    # ------------------------------------------------------------------
    # ΑΝΑΙΡΕΣΗ (CTRL+Z)
    # ------------------------------------------------------------------

    def _undo(self, event=None):
        """
        Αναίρεση της τελευταίας ενέργειας σχεδίασης ή καθαρισμού (Ctrl+Z).

        Επαναφέρει την PIL εικόνα παρασκηνίου στην τελευταία αποθηκευμένη
        κατάσταση από τη στοίβα αναίρεσης και ξανασχεδιάζει τον Tkinter καμβά.
        Αν η στοίβα είναι κενή, δεν κάνει τίποτα.
        """
        if not self.undo_stack:
            return

        # Επαναφορά PIL εικόνας στο τελευταίο snapshot
        self.pil_image = self.undo_stack.pop()
        self.draw = ImageDraw.Draw(self.pil_image)

        # Ξανασχεδίαση του Tkinter καμβά από την PIL εικόνα
        self.canvas.delete("all")
        photo = ImageTk.PhotoImage(self.pil_image)
        self.canvas.create_image(0, 0, anchor="nw", image=photo)
        self.canvas._undo_photo = photo  # αποφυγή garbage collection

    # ------------------------------------------------------------------
    # ΠΡΟΕΠΕΞΕΡΓΑΣΙΑ ΕΙΚΟΝΑΣ
    # ------------------------------------------------------------------

    def _preprocess(self) -> tuple[Image.Image, np.ndarray] | None:
        """
        Προεπεξεργασία της εικόνας του καμβά σε διάνυσμα 256 στοιχείων.

        Σε αντίθεση με το απλό resize, εδώ εντοπίζουμε πρώτα το bounding box
        της ζωγραφιάς, το κεντράρουμε σε τετράγωνο καμβά με padding, και μόνο
        τότε κλιμακώνουμε σε 16×16. Αυτό εξασφαλίζει ότι το ίδιο ψηφίο
        δίνει την ίδια αναπαράσταση ανεξάρτητα από πού το σχεδίασε ο χρήστης
        στον καμβά — ακριβώς όπως έγινε και η κανονικοποίηση του dataset.

        Βήματα:
            1. getbbox() → εύρεση ορθογωνίου περιοχής με μελάνι.
            2. Crop στο bounding box.
            3. Τοποθέτηση σε τετράγωνο με ανάλογο padding (~20% της μεγαλύτερης
               πλευράς) ώστε το ψηφίο να μην «αγγίζει» τα άκρα.
            4. GaussianBlur για μαλάκωση ακμών πριν το downsample.
            5. Resize σε 16×16 με LANCZOS.
            6. Επιστροφή (PIL 16×16 εικόνα, float64 ndarray 256 στοιχείων).

        Returns
        -------
        tuple[Image.Image, np.ndarray] | None
            None αν ο καμβάς είναι κενός (getbbox() επιστρέφει None).
        """
        # Βρες το tight bounding box των λευκών pixels
        bbox = self.pil_image.getbbox()
        if bbox is None:
            return None  # κενός καμβάς

        # Κόψε ακριβώς το περιεχόμενο
        cropped = self.pil_image.crop(bbox)
        cw, ch = cropped.size

        # Padding: ~20% της μεγαλύτερης διάστασης, τουλάχιστον 2px
        # Αυτό αντιστοιχεί στο "margin" που έχουν τα MNIST δεδομένα γύρω
        # από κάθε ψηφίο μετά τη στρέβλωση/κεντράρισμά τους.
        pad = max(2, max(cw, ch) // 5)

        # Δημιούργησε τετράγωνο καμβά (μαύρο φόντο) και κέντρα το crop
        side = max(cw, ch) + 2 * pad
        square = Image.new("L", (side, side), 0)
        ox = (side - cw) // 2
        oy = (side - ch) // 2
        square.paste(cropped, (ox, oy))

        # Gaussian blur πριν το downsample (μαλακώνει ακμές όπως το dataset)
        blurred = square.filter(ImageFilter.GaussianBlur(radius=1))

        # Resize σε 16×16 με LANCZOS
        small = blurred.resize(MODEL_IMG_SIZE, Image.LANCZOS)
        arr = np.array(small, dtype=np.float64).flatten() / 255.0
        return small, arr

    # ------------------------------------------------------------------
    # LIVE PREVIEW (ενημέρωση μετά από κάθε stroke)
    # ------------------------------------------------------------------

    def _update_live_preview(self):
        """
        Ενημέρωση του preview label αμέσως μετά από κάθε stroke (mouse release).

        Δείχνει στον χρήστη σε πραγματικό χρόνο τι «βλέπει» το μοντέλο μετά
        το centering και το resize — χωρίς να εκτελεί πρόβλεψη. Αν ο καμβάς
        είναι κενός, το preview μένει κενό.
        """
        result = self._preprocess()
        if result is None:
            return
        small, _ = result
        preview_img = small.resize(PREVIEW_SIZE, Image.NEAREST)
        preview_photo = ImageTk.PhotoImage(preview_img)
        self.preview_label.config(image=preview_photo, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1])
        self.preview_label.image = preview_photo

    # ------------------------------------------------------------------
    # ΠΡΟΒΛΕΨΗ
    # ------------------------------------------------------------------

    def _predict(self):
        """
        Εκτέλεση πρόβλεψης για την εικόνα που έχει σχεδιαστεί στον καμβά.

        Βήματα:
            1. Έλεγχος ότι έχει φορτωθεί μοντέλο.
            2. Προεπεξεργασία μέσω _preprocess() (centering + blur + resize).
            3. Έλεγχος κενού καμβά.
            4. Κλήση predict_digit_by_relative_residual() για πρόβλεψη.
            5. Εμφάνιση αποτελέσματος και ενημέρωση preview.
        """
        # Έλεγχος ότι έχει φορτωθεί μοντέλο πριν από οποιαδήποτε πρόβλεψη
        if self.digit_u_dict is None:
            self.result_label.config(text="❌", fg="red")
            self.detail_label.config(text="Φόρτωσε πρώτα ένα .pkl αρχείο.")
            return

        # ── Βήμα 1: Προεπεξεργασία (centering + blur + resize) ─────────────
        result = self._preprocess()
        if result is None:
            # Κενός καμβάς — εμφάνιση μηνύματος αντί για inf residuals
            self.result_label.config(text="✏️", fg="#888888")
            self.detail_label.config(text="Ζωγράφισε πρώτα ένα ψηφίο στον καμβά.")
            return
        small, arr = result

        # ── Βήμα 2: Πρόβλεψη ────────────────────────────────────────────────
        predicted = predict_digit_by_relative_residual(arr, self.digit_u_dict)

        # ── Βήμα 3: Ενημέρωση αποτελέσματος ─────────────────────────────────
        self.result_label.config(text=str(predicted), fg="#1a73e8")

        # ── Βήμα 4: Ενημέρωση preview ────────────────────────────────────────
        # Μεγεθύνουμε το 16×16 σε PREVIEW_SIZE με NEAREST για να φαίνονται
        # ευδιάκριτα τα pixels χωρίς θόλωμα (nearest-neighbor interpolation)
        preview_img = small.resize(PREVIEW_SIZE, Image.NEAREST)
        preview_photo = ImageTk.PhotoImage(preview_img)
        self.preview_label.config(image=preview_photo, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1])
        self.preview_label.image = preview_photo  # αποφυγή garbage collection

        # ── Βήμα 5: Υπολογισμός και εμφάνιση residuals για κάθε ψηφίο ───────
        # Υπολογίζουμε το σχετικό υπόλειμμα για κάθε ψηφίο ώστε να δείξουμε
        # στον χρήστη πόσο "σίγουρη" είναι η πρόβλεψη σε σχέση με τα άλλα ψηφία
        detail_lines = []
        for d in range(10):
            U = self.digit_u_dict[d]["U_cols"]                   # (256, k)
            projection = U @ (U.T @ arr)                          # προβολή στον υπόχωρο
            residual = np.linalg.norm(arr - projection)           # υπόλειμμα
            norm = np.linalg.norm(arr)                            # νόρμα της εικόνας
            relative = residual / norm if norm > 1e-10 else float("inf")
            marker = "  ◀" if d == predicted else ""
            detail_lines.append(f"  {d}:  {relative:.4f}{marker}")

        self.detail_label.config(text="\n".join(detail_lines))


# ----------------------------------------------------------------------------------
# ΣΗΜΕΙΟ ΕΙΣΟΔΟΥ
# ----------------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    DigitPredictor(root)
    root.mainloop()