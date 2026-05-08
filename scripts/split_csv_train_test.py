import pandas as pd
import numpy as np

df = pd.read_csv("data/data_for_train_m_n.csv")

wave_ids = df["wave_id"].unique()

rng = np.random.default_rng(42)
rng.shuffle(wave_ids)

split = int(len(wave_ids) * 0.8)

train_ids = wave_ids[:split]
test_ids = wave_ids[split:]

train_df = df[df["wave_id"].isin(train_ids)]
test_df = df[df["wave_id"].isin(test_ids)]

train_df.to_csv("data/train_split.csv", index=False)
test_df.to_csv("data/test_split.csv", index=False)

print("train:", len(train_ids), "test:", len(test_ids))