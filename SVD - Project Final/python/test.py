from functions import *

if __name__ == "__main__":
    optimized_files = sorted(
        # from the "trained_models" directory, find all files that match the pattern "trained_u_dict_optimized_*.pkl"
        Path("trained_models").glob("trained_u_dict_optimized_*.pkl"),
        key=lambda p: float(p.stem.split("_")[-1]),
    )

    for optimized_path in optimized_files:
        with open(optimized_path, "rb") as file:
            restored_data = pickle.load(file)

        print(f"\nEvaluating: {optimized_path.name}")
        confusion_matrix, overall_accuracy = evaluate_on_test_set(
            digit_u_dict=restored_data,
            file_path="data.xlsx",
            set_type="test",
        )
        print_results(confusion_matrix, overall_accuracy)