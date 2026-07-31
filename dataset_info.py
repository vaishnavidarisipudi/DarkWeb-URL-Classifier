import pandas as pd

print("Loading Dataset...")

df = pd.read_csv("dataset/malicious_phish.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nNumber of samples in each class:")
print(df["type"].value_counts())