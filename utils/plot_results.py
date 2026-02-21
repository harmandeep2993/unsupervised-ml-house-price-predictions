import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Plot the resuls of current step
def plot_results(df_results, step):

    # Sort properly (descending is better for performance comparison)
    results_sorted = df_results.sort_values(by='overfit_gap', ascending=True)

    models = results_sorted["model_name"]
    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8, 6))

    bars1 = plt.bar(x - width/2, results_sorted["r2_train"], width, label="R2 Train", color= "#E76F51" )
    bars2 = plt.bar(x + width/2, results_sorted["r2_test"],  width, label="R2 Test", color="#2A9D8F")

    plt.xticks(x, models, rotation=40, ha='right')
    plt.ylabel("R2 Score")
    plt.title(f"{step.capitalize()} Model Comparison", fontsize= 12)
    plt.ylim(0, 1.15)

    # Add Overfit Gap labels
    for i, gap in enumerate(results_sorted["overfit_gap"]):
        max_height = max(results_sorted["r2_train"].iloc[i],
                        results_sorted["r2_test"].iloc[i])
        
        plt.text(x[i],
                max_height + 0.03,
                f"Gap: {gap:.1f}%",
                ha='center',
                fontsize=8)

    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()

# Plot the results between two different steps such as scaled dataset vs unscaled dataset
def scale_unscale_model_comparison(df1, df2):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, df, title in zip(
        axes,
        [df1, df2],
        ["Baseline", "Baseline Scaled"]
    ):
        
        df_sorted = df.sort_values(by="model_name")
        
        models = df_sorted["model_name"]
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df_sorted["r2_train"], width, label="Train", color= "#E76F51")
        bars2 = ax.bar(x + width/2, df_sorted["r2_test"],  width, label="Test",color= "#2A9D8F")
        
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.set_title(title)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Overfit gap label
        for i, gap in enumerate(df_sorted["overfit_gap"]):
            max_height = max(df_sorted["r2_train"].iloc[i],
                            df_sorted["r2_test"].iloc[i])
            ax.text(x[i], max_height + 0.03,
                    f"OG: {gap:.1f}%",
                    ha='center',
                    fontsize=8)

    axes[0].set_ylabel("R² Score")
    axes[1].legend()

    plt.suptitle("Model Comparison: Baseline vs Baseline-Scaled")
    plt.tight_layout()
    plt.show()