import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set style for plots
sns.set(style="whitegrid")

# === 1. Load Dataset ===
file_path = 'D:/titanic-data-cleaning/data/Titanic-Dataset.csv'
df = pd.read_csv(file_path)

print("✅ Data loaded successfully.")
print("\n📊 Dataset Preview:")
print(df.head())

# === 2. Summary Statistics ===
print("\n📈 Summary Statistics:")
print(df.describe(include='all'))

# === 3. Histograms for Numeric Features ===
numeric_cols = df.select_dtypes(include='number').columns
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'output/hist_{col}.png')
    plt.close()

print("📌 Histograms saved to output/ folder.")

# === 4. Boxplots to Check Outliers ===
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.savefig(f'output/box_{col}.png')
    plt.close()

print("📌 Boxplots saved to output/ folder.")

# === 5. Correlation Heatmap ===
plt.figure(figsize=(10, 8))
corr = df[numeric_cols].corr() # type: ignore
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('output/correlation_heatmap.png')
plt.close()

print("📌 Correlation heatmap saved.")

# === 6. Pairplot (Optional, can be slow) ===
# sns.pairplot(df[numeric_cols])
# plt.savefig('output/pairplot.png')
# plt.close()
# print("📌 Pairplot saved (disabled by default).")

print("\n✅ EDA complete. All visuals saved in /output folder.")
