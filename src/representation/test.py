"""
Compare your run (JSON metrics) against the screenshot table (paper baselines).

Primary metric: F1 (default: macro_f1 from your JSON).
Outputs:
  1) Summary: % worse / % better / ties
  2) A comparison table (all models)
  3) A filtered table showing ONLY models with better F1 than your run
  4) (Optional) LaTeX table for the better-than-your-run models
"""

from dataclasses import dataclass
import pandas as pd

# ----------------------------
# 1) Your results (paste/keep)
# ----------------------------
my_results = {
  "metrics": {
    "accuracy": 0.6915584415584416,
    "macro_precision": 0.6516757885557583,
    "macro_recall": 0.49421679107601174,
    "macro_f1": 0.4865844593117321,
    "weighted_precision": 0.6949876996433053,
    "weighted_recall": 0.6915584415584416,
    "weighted_f1": 0.6200927303052451,
  }
}

# Choose which F1 you want to compare against the screenshot.
# The screenshot table's "F1" is typically macro-F1; keep macro_f1 unless you KNOW it should be weighted_f1.
F1_KEY = "macro_f1"
my_f1 = float(my_results["metrics"][F1_KEY])

# -----------------------------------------
# 2) Screenshot baselines (transcribed list)
# -----------------------------------------
# Columns in screenshot: Acc, Prec, Recall, F1. We only need F1 for this comparison.
baselines = [
    # Prompting — direct clarity
    {"group": "Prompting", "variant": "direct clarity",        "model": "ZS Llama-70b",      "f1": 0.259},
    {"group": "Prompting", "variant": "direct clarity",        "model": "ZS Falcon-40b",     "f1": 0.144},
    {"group": "Prompting", "variant": "direct clarity",        "model": "ZS ChatGPT",        "f1": 0.413},
    {"group": "Prompting", "variant": "direct clarity",        "model": "FS Llama-7b",       "f1": 0.219},
    {"group": "Prompting", "variant": "direct clarity",        "model": "FS Llama-13b",      "f1": 0.156},
    {"group": "Prompting", "variant": "direct clarity",        "model": "FS Llama-70b",      "f1": 0.333},
    {"group": "Prompting", "variant": "direct clarity",        "model": "FS Falcon-7b",      "f1": 0.152},
    {"group": "Prompting", "variant": "direct clarity",        "model": "FS Falcon-40b",     "f1": 0.186},
    {"group": "Prompting", "variant": "direct clarity",        "model": "standalone CoT",    "f1": 0.368},

    # Prompting — evasion-based clarity
    {"group": "Prompting", "variant": "evasion-based clarity", "model": "ZS Llama-70b",      "f1": 0.261},
    {"group": "Prompting", "variant": "evasion-based clarity", "model": "ZS Falcon-40b",     "f1": 0.375},
    {"group": "Prompting", "variant": "evasion-based clarity", "model": "ZS ChatGPT",        "f1": 0.482},
    {"group": "Prompting", "variant": "evasion-based clarity", "model": "FS Llama-7b",       "f1": 0.262},
    {"group": "Prompting", "variant": "evasion-based clarity", "model": "FS Llama-13b",      "f1": 0.259},
    {"group": "Prompting", "variant": "evasion-based clarity", "model": "FS Llama-70b",      "f1": 0.365},
    {"group": "Prompting", "variant": "evasion-based clarity", "model": "FS Falcon-7b",      "f1": 0.222},
    {"group": "Prompting", "variant": "evasion-based clarity", "model": "FS Falcon-40b",     "f1": 0.200},
    {"group": "Prompting", "variant": "evasion-based clarity", "model": "standalone CoT",    "f1": 0.510},
    {"group": "Prompting", "variant": "evasion-based clarity", "model": "multi CoT",         "f1": 0.462},

    # Tuned models — direct clarity
    {"group": "Tuned models", "variant": "direct clarity",     "model": "DeBERTa-base",      "f1": 0.441},
    {"group": "Tuned models", "variant": "direct clarity",     "model": "RoBERTa-base",      "f1": 0.530},
    {"group": "Tuned models", "variant": "direct clarity",     "model": "XLNet-base",        "f1": 0.518},
    {"group": "Tuned models", "variant": "direct clarity",     "model": "Llama-7b",          "f1": 0.457},
    {"group": "Tuned models", "variant": "direct clarity",     "model": "Llama-13b",         "f1": 0.580},
    {"group": "Tuned models", "variant": "direct clarity",     "model": "Llama-70b",         "f1": 0.680},
    {"group": "Tuned models", "variant": "direct clarity",     "model": "Falcon-7b",         "f1": 0.175},
    {"group": "Tuned models", "variant": "direct clarity",     "model": "Falcon-40b",        "f1": 0.356},

    # Tuned models — evasion-based clarity
    {"group": "Tuned models", "variant": "evasion-based clarity","model": "DeBERTa-base",     "f1": 0.537},
    {"group": "Tuned models", "variant": "evasion-based clarity","model": "RoBERTa-base",     "f1": 0.495},
    {"group": "Tuned models", "variant": "evasion-based clarity","model": "XLNet-base",       "f1": 0.546},
    {"group": "Tuned models", "variant": "evasion-based clarity","model": "Llama-7b",         "f1": 0.616},
    {"group": "Tuned models", "variant": "evasion-based clarity","model": "Llama-13b",        "f1": 0.616},
    {"group": "Tuned models", "variant": "evasion-based clarity","model": "Llama-70b",        "f1": 0.682},
    {"group": "Tuned models", "variant": "evasion-based clarity","model": "Falcon-7b",        "f1": 0.397},
    {"group": "Tuned models", "variant": "evasion-based clarity","model": "Falcon-40b",       "f1": 0.558},
]

# ----------------------------
# 3) Build comparison table
# ----------------------------
df = pd.DataFrame(baselines)
df["my_f1"] = my_f1
df["delta"] = df["f1"] - my_f1
df["relation"] = df["delta"].apply(lambda d: "better" if d > 0 else ("worse" if d < 0 else "tie"))

total = len(df)
pct_worse  = 100.0 * (df["relation"] == "worse").mean()
pct_better = 100.0 * (df["relation"] == "better").mean()
pct_tie    = 100.0 * (df["relation"] == "tie").mean()

print(f"My {F1_KEY} = {my_f1:.6f}")
print(f"Total screenshot models: {total}")
print(f"% worse than my run:  {pct_worse:.1f}%")
print(f"% better than my run: {pct_better:.1f}%")
print(f"% ties:              {pct_tie:.1f}%\n")

# Full table (sorted by F1 desc)
df_sorted = df.sort_values(["f1", "group", "variant", "model"], ascending=[False, True, True, True])
print("=== All models (sorted by F1 desc) ===")
print(df_sorted[["group", "variant", "model", "f1", "delta", "relation"]].to_string(index=False,
      formatters={"f1": "{:.3f}".format, "delta": "{:+.3f}".format}))

# Better-only table
better = df_sorted[df_sorted["relation"] == "better"].copy()
print("\n=== Models with better F1 than my run ===")
if better.empty:
    print("None.")
else:
    print(better[["group", "variant", "model", "f1", "delta"]].to_string(index=False,
          formatters={"f1": "{:.3f}".format, "delta": "{:+.3f}".format}))

# ----------------------------
# 4) Optional: LaTeX table of better models
# ----------------------------
# Comment out if not needed.
if not better.empty:
    latex = better[["group", "variant", "model", "f1", "delta"]].to_latex(
        index=False,
        float_format="%.3f",
        caption=f"Models in the screenshot with higher F1 than my run (my {F1_KEY}={my_f1:.3f}).",
        label="tab:better_than_my_run",
        escape=True
    )
    print("\n=== LaTeX (better-than-my-run) ===")
    print(latex)
