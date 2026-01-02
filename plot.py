from typing import Dict, List
import os
import matplotlib.pyplot as plt


def load_epoch_results(
    results_dir: str,
    epochs: List[int],
) -> Dict[str, List[float]]:
    """
    Load per-epoch results from JSON files and return
    a dict of metric -> list of values ordered by epoch.
    """

    metrics = {
        "in_distribution_k3": [],
        "no_carry_k3": [],
        "single_carry_k3": [],
        "multi_carry_k3": [],
        "length_generalization_k4": [],
    }

    for epoch in epochs:
        path = os.path.join(results_dir, f"epoch_{epoch}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing results file: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        for key in metrics:
            if key not in data:
                raise KeyError(f"Key '{key}' missing in {path}")
            metrics[key].append(data[key])

    return metrics

results_dir = "results"
epochs = list(range(1, 11))

results = load_epoch_results(results_dir, epochs)

# Sanity check
print("Loaded results:")
for k, v in results.items():
    print(f"{k}: {v}")

id_acc = results["in_distribution_k3"]
multi_carry_acc = results["multi_carry_k3"]
length_gen_acc = results["length_generalization_k4"]

plt.figure(figsize=(6, 5))

plt.plot(epochs, results["in_distribution_k3"], marker="o", label="In distribution (k=3)")
plt.plot(epochs, results["no_carry_k3"], marker="o", label="No-carry (k=3)")
plt.plot(epochs, results["single_carry_k3"], marker="o", label="Single-carry (k=3)")
plt.plot(epochs, results["multi_carry_k3"], marker="o", label="Multi-carry (k=3)")
plt.plot(epochs, results["length_generalization_k4"], marker="o", label="Length generalization (k=4)")

plt.rcParams["font.size"] = 11
plt.rcParams['axes.titlesize'] = 11

plt.xlabel("Training epoch")
plt.ylabel("Exact-match accuracy")
plt.ylim(-0.01, 0.25)
plt.xticks(epochs)
plt.legend()
plt.tight_layout()
plt.show()