from datasets import Dataset
from collections import defaultdict, Counter
import random


class FinetuningDataset:

    def __init__(self, seed):
        self.seed = seed
        random.seed(seed)

    @staticmethod
    def count_carries(a: int, b: int) -> int:
        carry = 0
        carry_count = 0

        while a > 0 or b > 0:
            digit_sum = (a % 10) + (b % 10) + carry
            if digit_sum >= 10:
                carry = 1
                carry_count += 1
            else:
                carry = 0

            a //= 10
            b //= 10

        return carry_count

    def generate_example(self, k):
        a = random.randint(10**(k-1), 10**k - 1)
        b = random.randint(10**(k-1), 10**k - 1)
        s = a + b
        return {
            "prompt": f"{a} + {b} =",
            "ground_truth": str(s),
            "a": a,
            "b": b,
            "sum": s,
            "carry_count": self.count_carries(a, b),
        }

    def generate_split(self, n, k):
        data = defaultdict(list)
        for i in range(n):
            ex = self.generate_example(k)
            for key, value in ex.items():
                data[key].append(value)
            data["example_id"].append(i)
        return Dataset.from_dict(data)

    def generate_data(self, k=3, n_train=200_000, n_val=5_000, n_test=5_000):
        train_dataset = self.generate_split(n_train, k)
        val_dataset = self.generate_split(n_val, k)
        test_dataset = self.generate_split(n_test, k)
        return train_dataset, val_dataset, test_dataset

    def generate_carry_dataset(self, n, k, carry_type, seed = 0):
    # Generates a dataset of addition problems with a given number of digits k and a given carry type
    # carry_type from {"no_carry", "single_carry", "multi_carry"}
    # For evaluation purposes

        random.seed(seed)

        data = defaultdict(list)
        i = 0

        while len(data["prompt"]) < n:
            ex = self.generate_example(k)
            carry_count = ex["carry_count"]

            if carry_type == "no_carry" and carry_count != 0:
                continue
            if carry_type == "single_carry" and carry_count != 1:
                continue
            if carry_type == "multi_carry" and carry_count < 2:
                continue

            for key, value in ex.items():
                data[key].append(value)
            data["example_id"].append(i)
            i += 1

        return Dataset.from_dict(data)

    def eval_outputs(self, outputs):
        correct = 0
        total = len(outputs)
        all_predictions = []

        for out in outputs:
            pred = out["generated_text"].strip()
            gt = out["ground_truth"]

            is_correct = pred == gt
            correct += int(is_correct)

            all_predictions.append({
                "example_id": out["example_id"],
                "prediction": pred,
                "ground_truth": gt,
                "correct": is_correct,
                "prompt": out["prompt"]
            })

        accuracy = correct / total
        return accuracy, all_predictions

    @staticmethod
    def carry_histogram(dataset):
        return Counter(dataset["carry_count"])


def get_datasets(args, tokenizer):

    ds = FinetuningDataset(seed=args.seed)

    train_ds, val_ds, _ = ds.generate_data(
        k=args.data.k,
        n_train=args.data.n_train,
        n_val=args.data.n_val,
        n_test=args.data.n_test,
    )

    # SFTTrainer expects "prompt" and "completion"
    def format_for_sft(example):
        return {
            "prompt": example["prompt"],
            "completion": " " + example["ground_truth"],
            "ground_truth": example["ground_truth"],  # keep for eval
            "example_id": example["example_id"],
        }

    train_ds = train_ds.map(format_for_sft, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(format_for_sft, remove_columns=val_ds.column_names)

    return train_ds, val_ds
