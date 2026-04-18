import pickle
from pathlib import Path

from functions import (
    build_k_map_from_energy_threshold,
    build_u_dictionary_from_svd_cache,
    evaluate_on_dataset,
    evaluate_on_test_set,
    load_digit_dataset_from_excel,
    precompute_digit_svd_bases,
    print_results,
    split_train_validation_stratified,
    tune_k_map_coordinate_descent,
)


def run_optimized_pipeline(
    file_path="data.xlsx",
    train_set_type="zip",
    test_set_type="test",
    val_ratio=0.5,
    random_state=42,
    energy_threshold_init=0.9,
    k_min=1,
    k_max=45,
    max_passes=3,
    save_model=True,
    output_path="trained_models/trained_u_dict_tuned.pkl",
):
    """
    End-to-end pipeline:
    1) Split original training set into train_sub/validation.
    2) Initialize k per digit using energy threshold on train_sub.
    3) Tune per-digit k on validation set with coordinate descent.
    4) Retrain final subspaces on full training set using tuned k.
    5) Evaluate once on test set.
    """
    print("Step 1/5: Stratified split (train_sub / validation)")
    train_sub, val_set = split_train_validation_stratified(
        file_path=file_path,
        train_set_type=train_set_type,
        val_ratio=val_ratio,
        random_state=random_state,
    )

    print("Step 2/5: Initial k map from energy threshold")
    train_sub_svd = precompute_digit_svd_bases(train_sub)
    initial_k_map = build_k_map_from_energy_threshold(
        train_sub_svd,
        energy_threshold=energy_threshold_init,
    )
    print("Initial k map:", initial_k_map)

    print("Step 3/5: Tune per-digit k on validation set")
    tuned_k_map, best_val_acc, history, _ = tune_k_map_coordinate_descent(
        train_sub_by_digit=train_sub,
        val_by_digit=val_set,
        initial_k_map=initial_k_map,
        k_min=k_min,
        k_max=k_max,
        max_passes=max_passes,
        verbose=True,
    )

    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("Tuned k map:", tuned_k_map)

    # Optional: inspect val confusion at final tuned config
    tuned_val_u = build_u_dictionary_from_svd_cache(train_sub_svd, tuned_k_map)
    val_conf, val_acc = evaluate_on_dataset(tuned_val_u, val_set)
    print("\nValidation results (tuned model trained on train_sub):")
    print_results(val_conf, val_acc)

    print("Step 4/5: Retrain final model on full training set with tuned k")
    full_train = load_digit_dataset_from_excel(file_path=file_path, set_type=train_set_type)
    full_train_svd = precompute_digit_svd_bases(full_train)
    final_u_dict = build_u_dictionary_from_svd_cache(full_train_svd, tuned_k_map)

    if save_model:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(final_u_dict, f)
        print(f"Saved tuned model to: {out}")

    print("Step 5/5: Final evaluation on test set")
    test_conf, test_acc = evaluate_on_test_set(
        digit_u_dict=final_u_dict,
        file_path=file_path,
        set_type=test_set_type,
    )
    print_results(test_conf, test_acc)

    return {
        "initial_k_map": initial_k_map,
        "tuned_k_map": tuned_k_map,
        "best_val_accuracy": best_val_acc,
        "history": history,
        "test_accuracy": test_acc,
        "final_model": final_u_dict,
    }


if __name__ == "__main__":
    run_optimized_pipeline(
        file_path="data.xlsx",
        train_set_type="zip",
        test_set_type="test",
        val_ratio=0.5,
        random_state=42,
        energy_threshold_init=0.9,
        k_min=1,
        k_max=45,
        max_passes=3,
        save_model=True,
        output_path="trained_models/trained_u_dict_tuned.pkl",
    )
