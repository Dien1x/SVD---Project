# ===== Τρίτη προσπάθεια κατασκευής μοντέλου αναγνώρισης μοτίβων =====

from core.main_body import *
from core.evaluation import *
from _03_optimized_k_train import *

import pickle

# Φόρτωση του dataset και διαχωρισμός σε training και validation set με stratified split
training_dataset, validation_dataset = split_train_validation_stratified(file_path="data.xlsx", train_set_type="zip", val_ratio=0.2, random_state=42)

# Προεπεξεργασία για την κατασκευή των SVD βάσεων για κάθε ψηφίο στο training set
svd_cache = precompute_digit_svd_bases(training_dataset)

# Αρχικός υπολογισμός του initial_k_map με βάση το energy threshold στο training set
initial_k_map = build_k_map_from_energy_threshold(svd_cache, energy_threshold=0.92)
print("Initial k map from energy threshold:", initial_k_map)

# Βελτιστοποίηση του k για κάθε ψηφίο με coordinate descent στο validation set
tuned_k_map, best_val_acc, history, _ = tune_k_map_coordinate_descent(
        svd_cache=svd_cache,
        validation_set=validation_dataset,
        initial_k_map=initial_k_map,
        k_min=1,
        k_max=25,
        max_passes=5,
        verbose=True,
    )

print("Tuned k map after coordinate descent:", tuned_k_map)
print(f"Best validation accuracy achieved: {best_val_acc:.2f}%")

# Κατασκευή του tuned_u_dict με βάση το tuned_k_map
tuned_u_dict = build_u_dictionary_from_svd_cache(svd_cache, tuned_k_map)

# Αποθήκευση του tuned_u_dict και του tuned_k_map σε αρχεία για μελλοντική χρήση
with open("trained_models/tuned_u_dict.pkl", "wb") as file:
    pickle.dump(tuned_u_dict, file)


# ανακτηση του tuned_u_dict από το αρχείο για επιβεβαίωση
with open("trained_models/tuned_u_dict.pkl", "rb") as file:
    restored_u_dict = pickle.load(file)

# αξιολόγηση του tuned μοντέλου στο test set
test_dataset = load_digit_dataset_from_excel(file_path="data.xlsx", set_type="test")
confusion_matrix, overall_accuracy = evaluate_on_dataset(restored_u_dict, test_dataset)
print_results(confusion_matrix, overall_accuracy)