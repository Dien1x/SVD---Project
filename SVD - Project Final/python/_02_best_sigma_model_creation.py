# ===== Δεύτερη προσπάθεια κατασκευής μοντέλου αναγνώρισης μοτίβων =====

from main_body import *
from evaluation import *
from _02_train_by_best_sigma import *

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

for i in range(70, 98, 1):
    # Δημιουργία του λεξικού digit_u_dict για το συγκεκριμένο energy_threshold
    u_dict = build_u_dictionary_from_svd_cache_for_sigma(svd_cache, energy_threshold=i/100)
    with open(os.path.join(trained_models_dir, f"trained_u_dict_optimized_{i/100:.2f}.pkl"), "wb") as file:
        pickle.dump(u_dict, file)

    print(f"Optimized Dictionary for threshold {i/100:.2f} has been pickled!")

optimized_files = sorted(
    # from the "trained_models" directory, find all files that match the pattern "trained_u_dict_optimized_*.pkl"
    Path(trained_models_dir).glob("trained_u_dict_optimized_*.pkl"),
    key=lambda p: float(p.stem.split("_")[-1]),
)

# Φόρτωση του test dataset για την αξιολόγηση των μοντέλων
test_dataset = load_digit_dataset_from_excel(file_path=file_path, set_type="test")
total_overall_accuracy = 0.0
best_k_model = None

for optimized_path in optimized_files:
    with open(optimized_path, "rb") as file:
        restored_data = pickle.load(file)

    print(f"\nEvaluating: {optimized_path.name}")
    confusion_matrix, overall_accuracy = evaluate_on_dataset(restored_data, test_dataset)
    if overall_accuracy > total_overall_accuracy:
        total_overall_accuracy = overall_accuracy
        best_k_model = optimized_path.name
    print_results(confusion_matrix, overall_accuracy)

print(f"\nBest Overall Accuracy: {total_overall_accuracy}")
print(f"Best Model: {best_k_model}")