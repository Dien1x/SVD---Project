# ===== Τρίτη προσπάθεια για την επιλογή των πρώτων k στηλών του U  =====
# ===== Χρήση των καλύτερων τιμών του πίνακα S για την επιλογή του k =====
# ===== Έπειτα βελτιστοποίηση των k για κάθε ψηφίο ξεχωριστά, με validation =====


from main_body import *
from evaluation import *

# η ιδέα είναι ΄ότι μπορούμε να ξεκινήσουμε με μια αρχική εκτίμηση των k 
# για κάθε ψηφίο, βασισμένη σε ένα ενεργειακό κατώφλι (π.χ., 90% της συνολικής ενέργειας),
# και έπειτα να βελτιστοποιήσουμε περαιτέρω τα k για κάθε ψηφίο ξεχωριστά,
# χρησιμοποιώντας ένα validation set για να μετρήσουμε την ακρίβεια και να
# κάνουμε coordinate descent optimization.

# Μέχρι τώρα φτιάχναμε ένα λεξικό digit_u_dict που περιέχει για κάθε ψηφίο το k και τις πρώτες k στήλες του U,
# και οι στήλες αυτές παρέμεναν ίδιες κατα τη διάρκεια της εκπαίδευσης. 
# Τώρα θα φτιάξουμε ένα k_map που θα περιέχει μόνο τα k για κάθε ψηφίο, βασισμένα σε ένα threshold,
# και θα χρησιμοποιήσουμε αυτό το k_map για να φτιάξουμε το digit_u_dict από το svd_cache, ώστε να έχουμε την ευελιξία 
# υπολογίζουμε εκ νέου τις πρώτες k στήλες του U για κάθε ψηφίο, ανάλογα με το k που επιλέγουμε σε κάθε βήμα της βελτιστοποίησης.


def build_k_map_from_energy_threshold(svd_cache, energy_threshold=0.9):
    """
    Δημιουργία αρχικού k_map για κάθε ψηφίο, βασισμένο σε ένα ενεργειακό κατώφλι.

    Parameters 
    ----------
    svd_cache : dict
        digit -> {"U": U, "S": S, "V": Vt, "max_k": max_k}
    energy_threshold : float, optional
        Το ποσοστό της συνολικής ενέργειας που θέλουμε να διατηρήσουμε (π.χ., 0.9 για 90%).
    Returns
    -------
    k_map : dict
        digit -> k (int)
    """

    # έλεγχος αν το energy_threshold είναι στο (0, 1]
    if not (0 < energy_threshold <= 1):
        raise ValueError("energy_threshold must be in (0, 1].")

    k_map = {}

    # εκτελούμε την ίδια ακριβώς διαδικασία που εκτελέσαμε και στο select_u_columns_by_sigma 
    # για κάθε ψηφίο, αλλά τώρα αποθηκεύουμε μόνο το k στο k_map
    for digit in range(10):
        if digit not in svd_cache:
            raise ValueError(f"svd_cache is missing digit key {digit}.")

        S = svd_cache[digit]["S"]
        sigma_energy = S ** 2
        cumulative_energy = np.cumsum(sigma_energy) / np.sum(sigma_energy)
        k = int(np.searchsorted(cumulative_energy, energy_threshold) + 1)

        max_k = int(svd_cache[digit]["max_k"])
        k_map[digit] = max(1, min(k, max_k))

    return k_map


def build_u_dictionary_from_svd_cache(svd_cache, k_map):
    """
    Δημιουργία του digit_u_dict με βάση το svd_cache και το k_map.

    Parameters
    ----------
    svd_cache : dict
        digit -> {"U": U, "S": S, "V": Vt, "max_k": max_k}
    k_map : dict
        digit -> k (int)

    Returns 
    -------
    digit_u_dict : dict
        digit -> {"k": k, "U_cols": U[:, :k]}
    """
    digit_u_dict = {}

    for digit in range(10):
        # έλεγχος αν το digit υπάρχει τόσο στο svd_cache όσο και στο k_map
        if digit not in svd_cache:
            raise ValueError(f"svd_cache is missing digit key {digit}.")
        # έλεγχος αν το digit υπάρχει στο k_map
        if digit not in k_map:
            raise ValueError(f"k_map is missing digit key {digit}.")

        # παίρνουμε τον πίνακα U και το max_k από το svd_cache για το συγκεκριμένο ψηφίο
        U = svd_cache[digit]["U"]
        max_k = int(svd_cache[digit]["max_k"])

        # παίρνουμε το k από το k_map για το συγκεκριμένο ψηφίο
        k = int(k_map[digit])

        # έλεγχος αν το k είναι θετικό και δεν υπερβαίνει το max_k
        if k < 1:
            raise ValueError(f"k for digit {digit} must be >= 1.")

        k = min(k, max_k)

        # αποθηκεύουμε στο digit_u_dict για το συγκεκριμένο ψηφίο το k και τις πρώτες k στήλες του U
        digit_u_dict[digit] = {"k": k, "U_cols": U[:, :k]}

    return digit_u_dict


def tune_k_map_coordinate_descent(
    svd_cache,
    validation_set,
    initial_k_map,
    k_min=1,
    k_max=45,
    max_passes=5,
    verbose=True,
):
    """
    Βελτιστοποίηση των k για κάθε ψηφίο ξεχωριστά, χρησιμοποιώντας coordinate descent optimization.

    Parameters
    ----------
    svd_cache : dict
        digit -> {"U": U, "S": S, "V": Vt, "max_k": max_k}
    validation_set : dict
        digit -> list of sample matrices (256, n_samples)
    initial_k_map : dict
        digit -> initial k (int) για κάθε ψηφίο, π.χ., βασισμένο σε ένα ενεργειακό κατώφλι
    k_min : int
        Το ελάχιστο k που θα δοκιμάσουμε για κάθε ψηφίο.
    k_max : int
        Το μέγιστο k που θα δοκιμάσουμε για κάθε ψηφίο.
    max_passes : int
        Ο μέγιστος αριθμός περασμάτων πάνω από όλα τα ψηφία για τη βελτιστοποίηση.
    verbose : bool
        Αν είναι True, εμφανίζει debug prints για την πρόοδο της βελτιστοποίησης.
    
    Returns
    -------
    best_k_map : dict
        digit -> best k (int) που βρέθηκε για κάθε ψηφίο μετά τη βελτιστοποίηση.
    best_accuracy : float
        Η καλύτερη ακρίβεια στο validation set που επιτεύχθηκε με το best
        k_map.
    history : list of dict
        Ιστορικό των αλλαγών στα k και των αντίστοιχων ακρίβειων στο validation set, για ανάλυση μετά το τέλος της βελτιστοποίησης.
    svd_cache : dict
        Επιστρέφεται το svd_cache για να μπορεί να χρησιμοποιηθεί μετά τη βελτιστοποίηση, αν χρειάζεται.
    """

    # έλεγχος αν το initial_k_map έχει όλα τα ψηφία από 0 έως 9
    if not all(digit in initial_k_map for digit in range(10)):
        raise ValueError("initial_k_map must contain keys for all digits 0-9.")
    
    # έλεγχος αν το k_min και k_max είναι λογικά
    if k_min < 1 or k_max < k_min:
        raise ValueError("k range is invalid. Require 1 <= k_min <= k_max.")


    # ξεκινάμε με το initial_k_map, αλλά το περιορίζουμε μέσα στα k_min και k_max και το max_k για κάθε ψηφίο
    best_k_map = {d: int(initial_k_map[d]) for d in range(10)}
    for d in range(10):
        max_allowed = min(k_max, int(svd_cache[d]["max_k"]))
        best_k_map[d] = max(k_min, min(best_k_map[d], max_allowed))

    # υπολογίζουμε την αρχική ακρίβεια στο validation set με το αρχικό best_k_map
    best_u_dict = build_u_dictionary_from_svd_cache(svd_cache, best_k_map)
    # υπολογίζουμε την ακρίβεια στο validation set με το αρχικό best_k_map
    _, best_accuracy = evaluate_on_dataset(best_u_dict, validation_set)

    # κρατάμε ιστορικό των αλλαγών που κάνουμε στα k και των αντίστοιχων ακρίβειων στο validation set, 
    # για να μπορούμε να αναλύσουμε τη διαδικασία μετά το τέλος της βελτιστοποίησης
    history = [
        {
            "pass": 0,
            "digit": None,
            "k_map": best_k_map.copy(),
            "accuracy": float(best_accuracy),
        }
    ]

    
    # κάνουμε μέχρι max_passes περάσματα πάνω από όλα τα ψηφία, 
    # προσπαθώντας να βελτιώσουμε το k για κάθε ψηφίο ξεχωριστά
    ## κ΄άθε πέρασμα αποτελεί μια πλήρη σάρωση όλων των ψηφίων, 
    ## διότι η βελτιστοποίηση μπορεί να επηρεάσει την ακρίβεια για όλα τα ψηφία, 
    ## και έτσι θέλουμε να δώσουμε την ευκαιρία σε κάθε ψηφίο να βελτιωθεί μετά από τις αλλαγές στα άλλα ψηφία
    for pass_idx in range(1, max_passes + 1):
        improved_in_pass = False

        for digit in range(10):
            # για κάθε ψηφίο, δοκιμάζουμε να αλλάξουμε το k σε ένα εύρος από k_min έως k_max,
            # αλλά περιορίζουμε το k_max από το max_k που έχουμε στο svd_cache για το συγκεκριμένο ψηφίο,
            # γιατί δεν μπορούμε να έχουμε περισσότερες στήλες του U από το max_k που μας δίνει το svd_cache
            max_allowed = min(k_max, int(svd_cache[digit]["max_k"]))
            if max_allowed < k_min:
                continue

            # ξεκινάμε με το καλύτερο k που έχουμε για το συγκεκριμένο 
            # ψηφίο και προσπαθούμε να το βελτιώσουμε
            local_best_k = best_k_map[digit]
            local_best_accuracy = best_accuracy

            for k_candidate in range(k_min, max_allowed + 1):
                # αν το k_candidate είναι το ίδιο με το τρέχον best k για το συγκεκριμένο ψηφίο,
                # δεν χρειάζεται να το δοκιμάσουμε ξανά, γιατί ήδη γνωρίζουμε
                # την ακρίβεια για αυτό το k από προηγούμενα περάσματα, 
                # οπότε το παραλείπουμε για να εξοικονομήσουμε χρόνο
                if k_candidate == best_k_map[digit]:
                    continue

                # δημιουργούμε ένα trial k_map όπου αλλάζουμε μόνο το k για το συγκεκριμένο ψηφίο στο k_candidate,
                # και κρατάμε τα υπόλοιπα k ίδια με το best_k_map,
                # γιατί θέλουμε να μετρήσουμε την επίδραση της αλλαγής του k μόνο για το συγκεκριμένο ψηφίο,
                # ενώ τα υπόλοιπα ψηφία παραμένουν στα καλύτερα k που έχουμε βρει μέχρι τώρα
                trial_k_map = best_k_map.copy()
                trial_k_map[digit] = k_candidate

                # δημιουργούμε το trial_u_dict με βάση το svd_cache και το trial_k_map,
                # και υπολογίζουμε την ακρίβεια στο validation set με αυτό το trial
                trial_u_dict = build_u_dictionary_from_svd_cache(svd_cache, trial_k_map)
                _, trial_accuracy = evaluate_on_dataset(trial_u_dict, validation_set)

                # αν η ακρίβεια με το trial k είναι καλύτερη από την καλύτερη ακρίβεια 
                # που έχουμε μέχρι τώρα για αυτό το ψηφίο, τότε ενημερώνουμε το local_best_k 
                # και το local_best_accuracy, για να κρατήσουμε την καλύτερη επιλογή του k 
                # για αυτό το ψηφίο μέσα σε αυτό το πέρασμα
                if trial_accuracy > local_best_accuracy:
                    local_best_accuracy = trial_accuracy
                    local_best_k = k_candidate

            if local_best_k != best_k_map[digit]:
                best_k_map[digit] = local_best_k
                best_accuracy = local_best_accuracy
                improved_in_pass = True
                history.append(
                    {
                        "pass": pass_idx,
                        "digit": digit,
                        "k_map": best_k_map.copy(),
                        "accuracy": float(best_accuracy),
                    }
                )
            # debug print για να βλέπουμε την πρόοδο της βελτιστοποίησης σε κάθε ψηφίο και κάθε πέρασμα
            if verbose:
                print(
                    f"Pass {pass_idx}: updated digit {digit} k -> {local_best_k}, "
                    f"val accuracy = {best_accuracy:.2f}%"
                )
        # debug print στο τέλος κάθε περάσματος για να βλέπουμε την συνολική πρόοδο μετά από κάθε πλήρη σάρωση όλων των ψηφίων
        if verbose:
            print(f"End of pass {pass_idx}, validation accuracy = {best_accuracy:.2f}%")

        # αν δεν υπήρξε καμία βελτίωση σε αυτό το πέρασμα, μπορούμε να σταματήσουμε νωρίτερα,
        # γιατί αυτό σημαίνει ότι έχουμε φτάσει σε ένα τοπικό μέγιστο 
        # και περαιτέρω περάσματα δεν θα βελτιώσουν την ακρίβεια
        if not improved_in_pass:
            # debug print για να δείξουμε ότι δεν υπήρξε καμία βελτίωση και ότι σταματάμε νωρίτερα
            if verbose:
                print("No improvement in this pass. Stopping early.")
            break

    return best_k_map, float(best_accuracy), history, svd_cache
