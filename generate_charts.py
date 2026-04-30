import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# File paths
input_file = r"C:\Users\Shruti Rajvanshi\Desktop\OpenAI Fellowship 2026\BharatAI Index\scoring_sheet.xlsx"
output_folder = r"C:\Users\Shruti Rajvanshi\Desktop\OpenAI Fellowship 2026\BharatAI Index\IMAGES"

# Create folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Load data
df = pd.read_excel(input_file)

# -----------------------------
# 1. Overall Model Performance
# -----------------------------
model_avg = df.groupby("Model")["Overall"].mean()

plt.figure()
model_avg.plot(kind="bar")
plt.title("Overall Model Performance")
plt.ylabel("Average Score")
plt.xlabel("Model")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "overall_model_performance.png"), dpi=300)
plt.close()

# -----------------------------
# 2. Dimension-wise Performance
# -----------------------------
dimensions = ["Language", "Culture", "Context", "Safety", "Bias"]
dim_avg = df.groupby("Model")[dimensions].mean()

plt.figure()
dim_avg.plot(kind="bar", figsize=(8,5))
plt.title("Model Performance Across Dimensions")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(title="Dimension")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "dimension_performance.png"), dpi=300)
plt.close()

# -----------------------------
# 3. Country-wise Performance
# -----------------------------
country_avg = df.groupby(["Country", "Model"])["Overall"].mean().unstack()

plt.figure()
country_avg.plot(kind="bar")
plt.title("Country-wise Model Performance")
plt.ylabel("Average Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "country_performance.png"), dpi=300)
plt.close()

# -----------------------------
# 4. Bias Analysis (B Prompts)
# -----------------------------
bias_df = df[df["Primary_Dimension"] == "Bias"]
bias_avg = bias_df.groupby(["Prompt_ID", "Model"])["Bias"].mean().unstack()

plt.figure()
plt.ylim(0,5)
bias_avg.plot(kind="bar")
plt.title("Bias Performance Across Prompts")
plt.ylabel("Bias Score")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "bias_analysis.png"), dpi=300)
plt.close()

# -----------------------------
# 5. Safety Consistency
# -----------------------------
safety_df = df[df["Primary_Dimension"] == "Safety"]

safety_avg = safety_df.groupby(["Country", "Model"])["Safety"].mean().unstack()

plt.figure()
safety_avg.plot(kind="bar")

plt.title("Safety Performance Across Countries")
plt.ylabel("Safety Score")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "safety_consistency.png"), dpi=300)
plt.close()

# -----------------------------
# 6. Heatmap (MOST IMPORTANT)
# -----------------------------
heatmap_data = df.groupby("Model")[["Language", "Culture", "Context", "Safety", "Bias"]].mean()

plt.figure()
sns.heatmap(heatmap_data, annot=True)
plt.title("Model vs Dimension Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "heatmap.png"), dpi=300)
plt.close()

print("All charts saved successfully in IMAGES folder.")