import pandas as pd
import matplotlib.pyplot as plt
from packaging import version
import numpy as np
# Učitaj Excel fajl
df = pd.read_excel("statistics_by_versions.xlsx", sheet_name="Visualization")

# Pretvori verziju u string (npr. '3.10'), i sortiraj po pravom verzionom redosledu
df["Version"] = df["Version"].astype(str)

# Filtriraj samo verzije 3.6 do 3.12
valid_versions = [f"3.{i}" for i in range(6, 13)]  # '3.6' to '3.12'
df = df[df["Version"].isin(valid_versions)]

# Filtriraj željene modele
models = ["Random Forest", "XGBoost", "LightGBM"]
df = df[df["Model"].isin(models)]

# Grupisanje po tipu karakteristika
df_all = df[df["Features"].str.lower().str.contains("all features")]
df_nokey = df[df["Features"].str.lower().str.contains("without keyword")]

df_all["Set"] = "All features"
df_nokey["Set"] = "Without keywords"

# Kombinacija i priprema grupa
df_combined = pd.concat([df_all, df_nokey])
df_combined["Group"] = df_combined["Version"] + " - " + df_combined["Model"]

# Sortiranje po verzijama i modelima
df_combined["VersionSortKey"] = df_combined["Version"].map(version.parse)
df_combined = df_combined.sort_values(by=["VersionSortKey", "Model"])
# Sortiranje po verzijama pravilno
def sort_versions(data):
    return data.sort_values(by="Version", key=lambda x: x.map(version.parse))

# Plot funkcija
def plot(data, title):
    data = sort_versions(data)
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ["Precision", "Recall", "F1"]
    linestyles = {"Precision": "-", "Recall": "--", "F1": ":"}
    colors = {"Random Forest": "blue", "XGBoost": "green", "LightGBM": "orange"}

    for model in data["Model"].unique():
        subset = data[data["Model"] == model]
        for metric in metrics:
            ax.plot(subset["Version"], subset[metric],
                    label=f"{model} - {metric}",
                    color=colors[model],
                    linestyle=linestyles[metric],
                    marker="o")

    ax.set_title(title)
    ax.set_xlabel("Version")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0.8, 1.0)
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



def plot_comparison(metric_name):
    groups = df_combined["Group"].unique()
    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    all_values = df_combined[df_combined["Set"] == "All features"].set_index("Group")[metric_name]
    nokey_values = df_combined[df_combined["Set"] == "Without keywords"].set_index("Group")[metric_name]

    all_values = all_values.reindex(groups)
    nokey_values = nokey_values.reindex(groups)

    ax.bar(x - width/2, all_values, width, label="All features")
    ax.bar(x + width/2, nokey_values, width, label="Without keywords")

    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} Comparison per Version + Model")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_ylim(0.75, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.show()

# Prikaz svih uporednih metrika
plot_comparison("Precision")
plot_comparison("Recall")
plot_comparison("F1")
# Prikaz grafika
plot(df_all, "All Features - Test Metrics (v3.6 to v3.12)")
plot(df_nokey, "Without Keywords - Test Metrics (v3.6 to v3.12)")
