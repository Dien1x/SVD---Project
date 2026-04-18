# === Πρόβλεψη και αξιολόγηση ===

import numpy as np
import pandas as pd

def predict_digit_by_relative_residual(unknown_image, digit_u_dict):
    """
    Πρόβλεψη του ψηφίου που αντιστοιχεί σε μια άγνωστη εικόνα,
    χρησιμοποιώντας τη μέθοδο της σχετικής υπολειμματικής απόστασης.

    Parameters
    ----------
    unknown_image : np.ndarray
        Input image as shape (256,) or (16,16).
    digit_u_dict : dict
        Output from build_digit_u_dictionary().

    Returns
    -------
    int
        Predicted digit in [0, 9].
    """
    if not digit_u_dict:
        raise ValueError("digit_u_dict is empty. Build/load a dictionary with entries for digits 0-9.")

    # σε περίπτωση που το unknown_image είναι 16x16, το μετατρέπουμε σε διάνυσμα 256
    x = np.asarray(unknown_image, dtype=float).reshape(-1)

    # έλεγχος αν το x έχει 256 στοιχεία
    if x.size != 256:
        raise ValueError(f"unknown_image must have 256 values, got {x.size}.")

    # .norm(x, ord=z) υπολογίζει την p-νορμά του x.
    # Εδώ χρησιμοποιούμε την 2-νορμά (Euclidean norm) για να υπολογίσουμε το μέτρο του διανύσματος x.
    x_norm = np.linalg.norm(x)
    if x_norm == 0:
        raise ValueError("unknown_image norm is zero; cannot compute relative residual.")

    best_digit = None
    # αρχικοποιούμε το best_score με άπειρο
    best_score = np.inf

    for digit, info in digit_u_dict.items():
        U = info["U_cols"]

        # θέλουμε να βρούμε το (I−UU^T)x = x - U(U^T x)
        # όπου U^T x υπολογίζει τους συντελεστές προβολής (coeffs) 
        # U(U^T x) υπολογίζει την προβολή του x στον χώρο που ορίζεται από τις στήλες του U. (x_proj)
        # αρα το reisidual είναι x - x_proj, και η σχετική υπολειμματική απόσταση είναι ||residual||2 / ||x||2.

        # coeffs = U.T @ x υπολογίζει τους συντελεστές προβολής του x πάνω στις στήλες του U.
        # @ είναι τελεστής της numpy για matrix multiplication.
        # ουσιαστικά βρίσκει εσωτερικά γινόμενο μεταξύ κάθε στήλης του U και του διανύσματος x, δίνοντας ένα διάνυσμα συντελεστών.
        coeffs = U.T @ x
        # x_proj = U @ coeffs υπολογίζει την προβολή του x στον χώρο που ορίζεται από τις στήλες του U.
        # πολλαπλασιάζοντας τις στήλες του U με τους αντίστοιχους συντελεστές, παίρνουμε την προσέγγιση του x στον υποχώρο που ορίζεται από τις στήλες του U.
        x_proj = U @ coeffs

        residual = x - x_proj

        relative_residual = np.linalg.norm(residual) / x_norm

        if relative_residual < best_score:
            best_score = relative_residual
            best_digit = digit

    return int(best_digit)


def evaluate_on_dataset(digit_u_dict, dataset_by_digit):
    """
    Αξιολόγηση της απόδοσης του ταξινομητή σε ένα σύνολο δεδομένων.

    Parameters
    ----------
    digit_u_dict : dict
        digit -> {"k": int, "U_cols": np.ndarray}
    dataset_by_digit : dict
        digit -> matrix (256, n_samples)

    Returns
    -------
    confusion : np.ndarray
        Shape (10, 10), rows=true digits, cols=predicted digits.
    accuracy : float
        Συνολική ακρίβεια σε ποσοστό.
    """
    # αρχικοποίηση πίνακα σύγχυσης 10x10 με μηδενικά
    confusion = np.zeros((10, 10), dtype=int)

    for true_digit in range(10):
        if true_digit not in dataset_by_digit:
            raise ValueError(f"dataset_by_digit is missing digit key {true_digit}.")

        # Xd είναι ο πίνακας δεδομένων για το συγκεκριμένο ψηφίο, με σχήμα (256, n_samples).
        Xd = np.asarray(dataset_by_digit[true_digit], dtype=float)

        # έλεγχος αν το Xd έχει 2 διαστάσεις και αν ο πρώτος άξονας έχει μέγεθος 256
        if Xd.ndim != 2 or Xd.shape[0] != 256:
            raise ValueError(f"Digit {true_digit} matrix must have shape (256, n_samples), got {Xd.shape}.")

        # για κάθε δείγμα (στήλη) του Xd, κάνουμε πρόβλεψη και ενημερώνουμε τον πίνακα σύγχυσης
        for j in range(Xd.shape[1]):
            sample = Xd[:, j]
            pred_digit = predict_digit_by_relative_residual(sample, digit_u_dict)
            confusion[true_digit, pred_digit] += 1

    total = confusion.sum()
    # .trace(confusion) επιστρέφει το άθροισμα των στοιχείων της κύριας διαγωνίου του πίνακα σύγχυσης, δηλαδή τον αριθμό των σωστών προβλέψεων.
    correct = np.trace(confusion)
    accuracy = (correct / total) * 100 if total > 0 else 0.0

    return confusion, accuracy

def print_results(confusion, accuracy):
    """
    Εκτυπώνει τον πίνακα σύγχυσης και την συνολική ακρίβεια.
    Parameters
    ----------
    confusion : np.ndarray
        Shape (10, 10), rows=true digits, cols=predicted digits.
    accuracy : float
        Συνολική ακρίβεια σε ποσοστό.
    """
    row_labels = [f"true_{d}" for d in range(10)]
    col_labels = [f"pred_{d}" for d in range(10)]

    df_conf = pd.DataFrame(confusion, index=row_labels, columns=col_labels)

    # σύνολο των δειγμάτων ανά πραγματικό ψηφίο (άθροισμα κάθε γραμμής του πίνακα σύγχυσης)
    row_totals = confusion.sum(axis=1)

    # υπολογισμός της ακρίβειας ανά ψηφίο ως ποσοστό των σωστών προβλέψεων επί το σύνολο των δειγμάτων για αυτό το ψηφίο
    # np.divide(a, b, out, where) εκτελεί στοιχειο-προς-στοιχείο διαίρεση a/b, 
    # με δυνατότητα να καθορίσουμε πού να γίνει η διαίρεση (where) 
    # και τι να βάλουμε στα στοιχεία όπου δεν γίνεται διαίρεση (out).

    # ουσιαστικά, έχουμε ένα διάνυσμα με όλα τα στοιχεία
    # της κύριας διαγωνίου του πίνακα σύγχυσης (δηλαδή τους σωστούς), 
    # και ένα διάνυσμα με το σύνολο των δειγμάτων για κάθε πραγματικό ψηφίο.
    # και κάνουμε τη διαίρεση μεταξύ τους για να πάρουμε την ακρίβεια ανά ψηφίο,
    # πολλαπλασιάζοντας επί 100 για να το έχουμε σε ποσοστό.
    per_digit_acc = np.divide(
        np.diag(confusion),
        row_totals,
        out=np.zeros_like(row_totals, dtype=float),
        where=row_totals != 0,
    ) * 100

    # προσθέτουμε μια στήλη "total" με το σύνολο των δειγμάτων για κάθε πραγματικό ψηφίο, 
    df_conf["total"] = row_totals
    # και μια στήλη "correct_%" με την ακρίβεια ανά ψηφίο.
    df_conf["correct_%"] = np.round(per_digit_acc, 2)

    print("Confusion table (rows=true digit, columns=predicted digit):")
    print(df_conf.to_string())
    print(f"\nOverall accuracy: {accuracy:.2f}%")