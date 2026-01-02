import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.dataset import FinetuningDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR_BASE = "exp/gpt2/grateful-sound-3/checkpoint-"

checkpoints = [3125,6250,9375,12500,15625,18750,21875,25000,28125,31250]

all_MODEL_DIR = [f"{MODEL_DIR_BASE}{c}" for c in checkpoints]

K_TRAIN = 3
K_TEST_LONG = 4
N_EVAL = 2000
MAX_NEW_TOKENS = 6   # enough for 3-digit addition

for i in range(10):

  MODEL_DIR = all_MODEL_DIR[i]

  print(f"Loading model from {MODEL_DIR}")
  tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
  model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(DEVICE)
  model.eval()

  ds = FinetuningDataset(seed=0)

  @torch.no_grad()
  def eval_dataset(dataset):
      correct = 0

      for ex in dataset:
          inputs = tokenizer(
              ex["prompt"],
              return_tensors="pt",
          ).to(DEVICE)

          output_ids = model.generate(
              **inputs,
              max_new_tokens=MAX_NEW_TOKENS,
              do_sample=False,
              pad_token_id=tokenizer.eos_token_id,
          )

          # Decode only the generated continuation
          gen = output_ids[0][inputs["input_ids"].shape[1]:]
          pred = tokenizer.decode(gen, skip_special_tokens=True).strip()

          if pred == ex["ground_truth"]:
              correct += 1

      return correct / len(dataset)

  print("Evaluating in-distribution (k=3)")
  _, _, test_id = ds.generate_data(k=K_TRAIN, n_test=N_EVAL)
  acc_id = eval_dataset(test_id)

  print("Evaluating carry-stratified splits")
  test_no = ds.generate_carry_dataset(N_EVAL, K_TRAIN, "no_carry")
  test_single = ds.generate_carry_dataset(N_EVAL, K_TRAIN, "single_carry")
  test_multi = ds.generate_carry_dataset(N_EVAL, K_TRAIN, "multi_carry")

  acc_no = eval_dataset(test_no)
  acc_single = eval_dataset(test_single)
  acc_multi = eval_dataset(test_multi)

  print("Evaluating length generalization (k=4)")
  test_k4 = ds.generate_split(N_EVAL, K_TEST_LONG)
  acc_k4 = eval_dataset(test_k4)

  results = {
      "in_distribution_k3": acc_id,
      "no_carry_k3": acc_no,
      "single_carry_k3": acc_single,
      "multi_carry_k3": acc_multi,
      "length_generalization_k4": acc_k4,
  }

  with open("results/epoch_"+str(i+1)+".json", "w") as f:
      json.dump(results, f, indent=2)

  print("\nFinal results:")
  for k, v in results.items():
      print(f"{k}: {v:.4f}")