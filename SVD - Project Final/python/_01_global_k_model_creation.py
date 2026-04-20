# ===== Πρώτη προσπάθεια κατασκευής μοντέλου αναγνώρισης μοτίβων =====

from main_body import *
from evaluation import *
from _01_train_by_global_k import *

import pickle
from pathlib import Path
import os

# Φόρτωση του dataset και προεπεξεργασία για την κατασκευή των SVD βάσεων
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "..", "data.xlsx")
trained_models_dir = os.path.join(script_dir, "..", "trained_models")

training_dataset = load_digit_dataset_from_excel(file_path=file_path, set_type="zip")
# Προεπεξεργασία για την κατασκευή των SVD βάσεων για κάθε ψηφίο
svd_cache = precompute_digit_svd_bases(training_dataset)

for i in range(2, 31, 1):
    # Δημιουργία του λεξικού digit_u_dict για το συγκεκριμένο k
    u_dict = build_u_dictionary_from_svd_cache_for_global_k(svd_cache, i)
    with open(os.path.join(trained_models_dir, f"trained_u_dict_k_{i}.pkl"), "wb") as file:
        pickle.dump(u_dict, file)

    print(f"Dictionary {i} has been pickled!")


k_files = sorted(
    # from the "trained_models" directory, find all files that match the pattern "trained_u_dict_k_*.pkl"
    Path(trained_models_dir).glob("trained_u_dict_k_*.pkl"),
    key=lambda p: int(p.stem.split("_")[-1]),
)

# Φόρτωση του test dataset για την αξιολόγηση των μοντέλων
test_dataset = load_digit_dataset_from_excel(file_path=file_path, set_type="test")
total_overall_accuracy = 0.0
best_k_model = None

for pkl_path in k_files:
    with open(pkl_path, "rb") as file:
        restored_data = pickle.load(file)

    print(f"\nEvaluating: {pkl_path.name}")
    confusion_matrix, overall_accuracy = evaluate_on_dataset(restored_data, test_dataset)
    if overall_accuracy > total_overall_accuracy:
        total_overall_accuracy = overall_accuracy
        best_k_model = pkl_path.name
    print_results(confusion_matrix, overall_accuracy)

print(f"\nBest Overall Accuracy: {total_overall_accuracy}")
print(f"Best Model: {best_k_model}")