# ===== Πρώτη προσπάθεια για την επιλογή των πρώτων k στηλών του U  =====
# ===== Χρήση του ίδιου k για όλα τα ψηφία (global k) και αξιολόγηση των μοντέλων =====


from core.main_body import *
import matplotlib.pyplot as plt

def select_u_columns_by_kappa(
        svd_cache,
        digit,
        k,
        show_sigma_plot=False,
    ):
    """
    Επιλογή των πρώτων k στηλών του U για ένα συγκεκριμένο ψηφίο, 
    με βάση την επιλογή του χρήστη.

    Parameters
    ----------
    svd_cache : dict
        digit -> {"U": U, "S": S, "V": Vt, "max_k": max_k}
    digit : int
        Το ψηφίο για το οποίο θέλουμε να επιλέξουμε τις στήλες του U (0-9).
    k : int
        Ο αριθμός των πρώτων στηλών του U που θέλουμε να επιλέξουμε.
    show_sigma_plot : bool, optional
        Αν True, εμφανίζει ένα γράφημα με τις τιμές των σιγματικών για το συγκεκριμένο ψηφίο και μια κάθετη γραμμή που δείχνει την επιλογή του k.
         (default is False)

    Returns
    -------
    k : int
        Ο αριθμός των στηλών του U που επιλέχθηκαν (μπορεί να είναι μικρότερος από το αρχικό k αν το max_k είναι μικρότερο).
    U_selected : np.ndarray
        Ο πίνακας U με τις επιλεγμένες k στήλες (256, k).
    """

    # έλεγχος αν το digit είναι έγκυρο
    if digit not in svd_cache:
        raise ValueError(f"Digit {digit} not found in svd_cache. Available digits: {list(svd_cache.keys())}.")
    
    # έλεγχος αν το k είναι θετικός ακέραιος
    if k < 1:
        raise ValueError(f"k must be >= 1. Received k={k}.")

    # έλεγχος αν το k είναι μεγαλύτερο από το max_k για το συγκεκριμένο ψηφίο
    if k > svd_cache[digit]["max_k"]:
        print(f"Warning: k={k} is greater than max_k={svd_cache[digit]['max_k']} for digit {digit}. Using max_k instead.")
        k = svd_cache[digit]["max_k"]
    
    # επιλογή των πρώτων k στηλών του U για το συγκεκριμένο ψηφίο
    U_selected = svd_cache[digit]["U"][:, :k]

    if show_sigma_plot:
        plt.figure(figsize=(6, 4))
        # arange(a,, b, z) δημιουργεί ένα array που ξεκινάει από a, 
        # αυξάνεται κατά z, και σταματάει πριν το b
        ## άξονας x: 1, 2, ..., αριθμός των σιγματικών μέχρι το μέγεθος του πίνακα S
        ## άξονας y: οι τιμές των στοιχείων του πίνακα S για το συγκεκριμένο ψηφίο
        plt.plot(np.arange(1, svd_cache[digit]["S"].shape[0] + 1), svd_cache[digit]["S"], marker="o", linewidth=1)
        plt.axvline(k, color="g", linestyle="--", label=f"k={k}")
        plt.title(f"Digit {digit} singular values")
        plt.xlabel("Index")
        plt.ylabel("Sigma")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return k, U_selected

def build_u_dictionary_from_svd_cache_for_global_k(svd_cache, k):
    """
    Δημιουργεί ένα λεξικό που αντιστοιχεί σε κάθε ψηφίο (0-9) τις πρώτες k στήλες του U από το svd_cache.
    Parameters
    ----------
    svd_cache : dict
        digit -> {"U": U, "S": S, "V": Vt, "max_k": max_k}
    k : int
        Ο αριθμός των πρώτων στηλών του U που θέλουμε να επιλέξουμε για κάθε ψηφίο.
    Returns
    -------
    digit_u_dict : dict
        digit -> {"k": k, "U_cols": U[:, :k]}
    """
    digit_u_dict = {}

    for digit in range(10):
        if digit not in svd_cache:
            raise ValueError(f"svd_cache is missing digit key {digit}.")

        
        k, U = select_u_columns_by_kappa(svd_cache, digit, k, show_sigma_plot=False)
        digit_u_dict[digit] = {"k": k, "U_cols": U[:, :k]}

    return digit_u_dict


if __name__ == "__main__":
    # Example usage
    file_path = "data.xlsx"
    set_type = "zip"
    dataset_by_digit = load_digit_dataset_from_excel(file_path, set_type)
    svd_cache = precompute_digit_svd_bases(dataset_by_digit)

    digit = 3
    k = 10
    k_selected, U_k = select_u_columns_by_kappa(svd_cache, digit, k, show_sigma_plot=True)
    print(f"Selected k for digit {digit}: {k_selected}")