# ====== Λειτουργίες για τη φόρτωση και προεπεξεργασία του dataset ======

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def load_digit_dataset_from_excel(file_path="data.xlsx", set_type="zip"):
    """
    Φόρτωση του dataset από το αρχείο Excel και οργάνωσή του σε ένα λεξικό 
    που αντιστοιχεί σε κάθε ψηφίο (0-9) τις αντίστοιχες στήλες του πίνακα X.

    Parameters:
        - file_path: Το μονοπάτι προς το αρχείο Excel που περιέχει τα δεδομένα.
        - set_type: Ένα string που καθορίζει τον τύπο του dataset (π.χ. "zip" ή "test").
                    Αυτό χρησιμοποιείται για να καθορίσει ποια φύλλα του Excel θα διαβάσει.
    Returns:
        - dataset: Ένα λεξικό όπου τα κλειδιά είναι τα ψηφία (0-9) 
        και οι τιμές είναι οι αντίστοιχες στήλες του πίνακα X 
        που αντιστοιχούν σε κάθε ψηφίο.
    """
    
    # φόρτωση δεδομένων απο το excel με την pandas
    df = pd.read_excel(file_path, sheet_name=f"a{set_type}", header=None)
    # με την μέθοδο .values μετατρέπουμε το DataFrame σε numpy array 
    # και με την μέθοδο .flatten() το μετατρέπουμε σε μονοδιάστατο πίνακα
    labels = pd.read_excel(file_path, sheet_name=f"d{set_type}", header=None).values.flatten()

    X = df.values  # (256, N)
    dataset = {}
    for digit in range(10):
        # δημιουργία μάσκας για το συγκεκριμένο ψηφίο
        # η μάσκα είναι ένας boolean πίνακας που έχει True στις θέσεις όπου το label είναι ίσο με το ψηφίο
        mask = labels == digit
        # επιλογή των στηλών του X για τις οποίες η μάσκα είναι True και αποθήκευση στο dataset
        dataset[digit] = X[:, mask]

    return dataset

def precompute_digit_svd_bases(dataset_by_digit):
    """
    Kατασκευή ενός λεξικού που αποθηκεύει τις SVD βάσεις για κάθε ψηφίο (0-9)

    Parameters
    ----------
    dataset_by_digit : dict
        digit -> matrix (256, n_samples)

    Returns
    -------
    svd_cache : dict
        digit -> {"U": U, "S": S, "V": Vt, "max_k": max_k}
    """
    svd_cache = {}

    for digit in range(10):
        # Έλεγχος για την ύπαρξη του ψηφίου στο dataset
        if digit not in dataset_by_digit:
            raise ValueError(f"dataset_by_digit is missing digit key {digit}.")

        # Μετατροπή των δεδομένων σε numpy array σε περίπτωση που δεν είναι ήδη
        # και έλεγχος ΄΄οτι τα στοιχεία είναι τύπου float γιατί
        # Το np.linalg.svd δουλεύει καλύτερα και ουσιαστικά προορίζεται για float.
        Xd = np.asarray(dataset_by_digit[digit], dtype=float)

        # Έλεγχος για το σχήμα του πίνακα 
        # πρέπει να έχει 256 γραμμές (features) και n_samples στήλες (δείγματα)
        ## .shape επιστρέφει ένα tuple με το σχήμα του πίνακα, 
        # όπου το πρώτο στοιχείο είναι ο αριθμός των γραμμών 
        # και το δεύτερο ο αριθμός των στηλών
        ## .ndim επιστρέφει τον αριθμό των διαστάσεων του πίνακα
        if Xd.ndim != 2 or Xd.shape[0] != 256:
            raise ValueError(
                f"Digit {digit} matrix must have shape (256, n_samples), got {Xd.shape}."
            )
        if Xd.shape[1] == 0:
            raise ValueError(f"Digit {digit} has no samples.")

        # εκτέλεση SVD στον πίνακα Xd για το συγκεκριμένο ψηφίο
        U, S, Vt = np.linalg.svd(Xd, full_matrices=False)
        svd_cache[digit] = {"U": U, "S": S, "V": Vt, "max_k": U.shape[1]}

    return svd_cache

def split_train_validation_stratified(
        file_path="data.xlsx",
        train_set_type="zip",
        val_ratio=0.5,
        random_state=42,
    ):
    """
    Διαχωρίζει το αρχικό training set σε δύο υποσύνολα: 
    ένα για εκπαίδευση (train_sub) και 
    ένα για validation (val_set),
    διατηρώντας την αναλογία των δειγμάτων για κάθε ψηφίο 
    (stratified split).

    Parameters
    ----------
    file_path : str
        Το μονοπάτι προς το αρχείο Excel που περιέχει τα δεδομένα.
    train_set_type : str
        Το τύπος του training set.
    val_ratio : float
        Η αναλογία των δειγμάτων που θα πάνε στο validation set.
    random_state : int
        Ο αριθμός γεννήτριας για τη δημιουργία τυχαίων αριθμών.


    Returns
    -------
    train_sub : dict
        digit -> matrix (256, n_train_sub_samples)
    val_set : dict
        digit -> matrix (256, n_val_samples)
    """

    # Έλεγχος για την εγκυρότητα του val_ratio
    if not (0 < val_ratio < 1):
        raise ValueError("val_ratio must be in (0, 1).")

    # Δημιουργία ενός τυχαίου αριθμού γεννήτριας με βάση το random_state για αναπαραγωγιμότητα
    rng = np.random.default_rng(random_state)
    # Φόρτωση του πλήρους training set για να κάνουμε τον διαχωρισμό
    full_train = load_digit_dataset_from_excel(file_path=file_path, set_type=train_set_type)

    train_sub = {}
    val_set = {}

    for digit in range(10):
        Xd = full_train[digit]  # (256, n)
        n = Xd.shape[1]
        if n == 0:
            raise ValueError(f"No samples found for digit {digit} in split '{train_set_type}'.")

        # Δημιουργία τυχαίας διάταξης των δεικτών των δειγμάτων για το συγκεκριμένο ψηφίο
        # Η μέθοδος .permutation() του rng δημιουργεί έναν array με μια τυχαία διάταξη των αριθμών από 0 έως n-1,
        # που αντιστοιχούν στους δείκτες των στηλών του Xd.
        perm = rng.permutation(n)
        # Υπολογισμός του αριθμού των δειγμάτων που θα πάνε στο validation set με βάση το val_ratio
        n_val = int(round(n * val_ratio))

        # Εάν υπάρχουν περισσότερα από 1 δείγματα, 
        # προσαρμόζουμε το n_val ώστε να μην είναι μεγαλύτερο από n-1,
        # για να διασφαλίσουμε ότι θα υπάρχει τουλάχιστον ένα δείγμα στο train_sub.
        if n > 1:
            n_val = max(1, min(n - 1, n_val))
        else:
            n_val = 1

        # Διαχωρισμός των δεικτών για το validation set και το train_sub
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        # Δημιουργία των υποσυνόλων για το συγκεκριμένο ψηφίο και αποθήκευση στα αντίστοιχα λεξικά
        # Εδώ επιλέγουμε τις στήλες του Xd που αντιστοιχούν στους δείκτες val_idx για το validation set
        # και στους δείκτες train_idx για το train_sub.
        train_sub[digit] = Xd[:, train_idx]
        val_set[digit] = Xd[:, val_idx]

    return train_sub, val_set