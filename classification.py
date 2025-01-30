import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Sample Data for Text Classification Models
data = {
    "Model": ["BERT", "DistilBERT", "RoBERTa", "XLNet", "ALBERT"],
    "Accuracy": [0.92, 0.89, 0.91, 0.90, 0.88],  
    "F1 Score": [0.91, 0.88, 0.90, 0.89, 0.87],  
    "Inference Time (s)": [0.85, 0.65, 0.78, 1.0, 0.55],  
    "Model Size (MB)": [420, 350, 500, 550, 300]  
}

# Step 1: Create DataFrame
df = pd.DataFrame(data)
print("Initial Data:\n", df)

# Step 2: Normalize Data Using MinMaxScaler
scaler = MinMaxScaler()

df_normalized = df.copy()
df_normalized["Accuracy"] = scaler.fit_transform(df[["Accuracy"]])
df_normalized["F1 Score"] = scaler.fit_transform(df[["F1 Score"]])
df_normalized["Inference Time (s)"] = scaler.fit_transform(-df[["Inference Time (s)"]])  # Lower is better
df_normalized["Model Size (MB)"] = scaler.fit_transform(-df[["Model Size (MB)"]])  # Lower is better

print("\nNormalized Data:\n", df_normalized)

# Step 3: Assign Weights (Sum should be 1)
weights = np.array([0.4, 0.3, 0.2, 0.1])  # Accuracy, F1 Score, Inference Time, Model Size

# Step 4: Calculate Weighted Normalized Matrix
weighted_matrix = df_normalized.iloc[:, 1:] * weights
print("\nWeighted Normalized Matrix:\n", weighted_matrix)

# Step 5: Calculate Ideal and Negative-Ideal Solutions
ideal_solution = weighted_matrix.max().values  # Best values
negative_ideal_solution = weighted_matrix.min().values  # Worst values

# Step 6: Calculate Euclidean Distances
dist_to_ideal = np.sqrt(((weighted_matrix - ideal_solution) ** 2).sum(axis=1))
dist_to_negative_ideal = np.sqrt(((weighted_matrix - negative_ideal_solution) ** 2).sum(axis=1))

# Step 7: Calculate Relative Closeness to Ideal Solution
relative_closeness = dist_to_negative_ideal / (dist_to_ideal + dist_to_negative_ideal)

# Add rankings to the DataFrame
df["TOPSIS Score"] = relative_closeness
df["Rank"] = df["TOPSIS Score"].rank(ascending=False)

print("\nFinal Rankings:\n", df)

# Save Results to CSV
df.to_csv("topsis_text_classification_results.csv", index=False)

# Step 8: Visualize Results
plt.barh(df["Model"], df["TOPSIS Score"], color="lightgreen")
plt.xlabel("TOPSIS Score")
plt.ylabel("Model")
plt.title("Model Rankings using TOPSIS for Text Classification")
plt.gca().invert_yaxis()

# Save the chart as a PNG file
plt.savefig("model_rankings_text_classification.png", dpi=300, bbox_inches="tight")

# Display the chart
plt.show()
