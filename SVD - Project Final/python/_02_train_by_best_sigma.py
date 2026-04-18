# ===== Δεύτερη προσπάθεια για την επιλογή των πρώτων k στηλών του U  =====
# ===== Χρήση των καλύτερων τιμών του πίνακα S για την επιλογή του k =====

from core.main_body import *
import matplotlib.pyplot as plt

def select_u_columns_by_sigma(
        svd_cache,
        digit,
        energy_threshold=0.9,
        show_sigma_plot=False,
    ):
    """
    Επιλογή των πρώτων k στηλών του U για ένα συγκεκριμένο ψηφίο, 
    με βάση την επιλογή του χρήστη για το ποσοστό της συνολικής ενέργειας που θέλει να διατηρήσει.

    Parameters
    ----------
    svd_cache : dict
        digit -> {"U": U, "S": S, "V": Vt, "max_k": max_k}
    digit : int
        Το ψηφίο για το οποίο θέλουμε να επιλέξουμε τις στήλες του U (0-9).
    energy_threshold : float, optional
        Το ποσοστό της συνολικής ενέργειας που θέλουμε να διατηρήσουμε (π.χ., 0.9 για 90%).
    show_sigma_plot : bool, optional
        Αν είναι True, εμφανίζει γράφημα των singular values και της αθροιστικής ενέργειας.
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

    # έλεγχος αν το energy_threshold είναι στο (0, 1]
    if not (0 < energy_threshold <= 1):
        raise ValueError("energy_threshold must be in (0, 1].")


    # s^2 αντιστοιχεί στο πόσο variance / πληροφορία εξηγεί κάθε component
    # χρησιμοποιούμε το S^2 για να έχουμε πιο αντιπροσωπευτική μέτρηση της "ενέργειας" που εξηγεί κάθε component
    # έτσι εμπλέκεται και η διασπορά της επιστήμης της ανάλυσης δεδομένων, όπου τα singular values 
    # σχετίζονται με την ποσότητα πληροφορίας που διατηρείται από κάθε component
    sigma_energy = svd_cache[digit]["S"] ** 2
    
    # .cumsum υπολογίζει το αθροιστικό άθροισμα των στοιχείων του sigma_energy,
    # cumulative_energy είναι ένα array που κάθε στοιχείο είναι το άθροισμα των σιγματικών τιμών μέχρι εκείνο το σημείο, 
    # διαιρεμένο με το συνολικό άθροισμα των τιμών του sigma_energy, δηλαδή την συνολική ενέργεια
    cumulative_energy = np.cumsum(sigma_energy) / np.sum(sigma_energy)

    # .searchsorted βρίσκει την θέση όπου θα έπρεπε να εισαχθεί το energy_threshold στο cumulative_energy για να διατηρηθεί η σειρά
    # πχ αν cumulative_energy = [0.5, 0.8, 0.95] και energy_threshold = 0.9, τότε searchsorted θα επιστρέψει 2, 
    # γιατί το 0.9 θα έπρεπε να εισαχθεί μετά το 0.8 και πριν το 0.95
    k = int(np.searchsorted(cumulative_energy, energy_threshold) + 1)

    # έλεγχος αν το k είναι μεγαλύτερο από το max_k για το συγκεκριμένο ψηφίο
    if k > svd_cache[digit]["max_k"]:
        print(f"Warning: k={k} is greater than max_k={svd_cache[digit]['max_k']} for digit {digit}. Using max_k instead.")
        k = svd_cache[digit]["max_k"]

    # επιλογή των πρώτων k στηλών του U για το συγκεκριμένο ψηφίο
    U_selected = svd_cache[digit]["U"][:, :k]

    if show_sigma_plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # first plot: singular values distribution
        axes[0].plot(np.arange(1, len(svd_cache[digit]["S"]) + 1), svd_cache[digit]["S"], marker="o", linewidth=1)
        axes[0].set_title(f"Digit {digit} singular values")
        axes[0].set_xlabel("Index")
        axes[0].set_ylabel("Sigma")
        axes[0].grid(alpha=0.3)

        # second plot: cumulative energy with threshold and selected k
        axes[1].plot(np.arange(1, len(svd_cache[digit]["S"]) + 1), cumulative_energy, marker="o", linewidth=1)
        axes[1].axhline(energy_threshold, color="r", linestyle="--", label=f"threshold={energy_threshold}")
        axes[1].axvline(k, color="g", linestyle="--", label=f"k={k}")
        axes[1].set_title("Cumulative energy")
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("Energy ratio")
        axes[1].set_ylim(0, 1.02)
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    return k, U_selected

def build_u_dictionary_from_svd_cache_for_sigma(svd_cache, energy_threshold):
    """
    Δημιουργεί ένα λεξικό που αντιστοιχεί σε κάθε ψηφίο (0-9) τις πρώτες k στήλες του U από το svd_cache, 
    όπου το k επιλέγεται με βάση το energy_threshold για κάθε ψηφίο.

    Parameters
    ----------
    svd_cache : dict
        digit -> {"U": U, "S": S, "V": Vt, "max_k": max_k}
    energy_threshold : float
        Το ποσοστό της συνολικής ενέργειας που θέλουμε να διατηρήσουμε (π.χ., 0.9 για 90%).

    Returns
    -------
    digit_u_dict : dict
        digit -> {"k": k, "U_cols": U[:, :k]}
    """
    digit_u_dict = {}

    for digit in range(10):
        if digit not in svd_cache:
            raise ValueError(f"svd_cache is missing digit key {digit}.")

        k, U = select_u_columns_by_sigma(svd_cache, digit, energy_threshold=energy_threshold, show_sigma_plot=False)
        digit_u_dict[digit] = {"k": k, "U_cols": U[:, :k]}

    return digit_u_dict


if __name__ == "__main__":
    # Example usage
    file_path = "data.xlsx"
    set_type = "zip"
    dataset_by_digit = load_digit_dataset_from_excel(file_path, set_type)
    svd_cache = precompute_digit_svd_bases(dataset_by_digit)

    digit = 3
    threshold = 0.92
    k_selected, U_k = select_u_columns_by_sigma(svd_cache, digit, energy_threshold=threshold, show_sigma_plot=True)
    print(f"Selected k for digit {digit}: {k_selected}")