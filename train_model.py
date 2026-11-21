
import os
import glob
import warnings
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# SETTINGS
# ==========================================

DATASET_PATH = "dataset"
MODEL_PATH = "models/nids_model.pkl"

CHUNK_SIZE = 50000
MAX_SAMPLES_PER_CLASS = 5000
SAMPLE_PER_CHUNK_PER_CLASS = 1000
RANDOM_STATE = 42


# ==========================================
# FIND CSV FILES
# ==========================================

csv_files = glob.glob(
    os.path.join(DATASET_PATH, "**", "*.csv"),
    recursive=True
)

print("\n===================================")
print("AUTOMATED NIDS - IMPROVED TRAINING")
print("===================================\n")

print("CSV files found:", len(csv_files))

if not csv_files:
    raise SystemExit("ERROR: No CSV files found in dataset folder.")


# ==========================================
# RANDOM CHUNK-BASED CLASS SAMPLING
# ==========================================

samples_by_class = {}

for file in csv_files:

    print("\nReading:", os.path.basename(file))

    try:

        for chunk in pd.read_csv(
            file,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ):

            chunk.columns = chunk.columns.astype(str).str.strip()

            if "Label" not in chunk.columns:
                print("Label column not found. Skipping file.")
                break

            chunk["Label"] = (
                chunk["Label"]
                .astype(str)
                .str.strip()
            )

            chunk = chunk.replace(
                [np.inf, -np.inf],
                np.nan
            )

            chunk = chunk.dropna(subset=["Label"])

            for label, group in chunk.groupby("Label"):

                if len(group) > SAMPLE_PER_CHUNK_PER_CLASS:

                    group = group.sample(
                        SAMPLE_PER_CHUNK_PER_CLASS,
                        random_state=RANDOM_STATE
                    )

                if label not in samples_by_class:
                    samples_by_class[label] = group.copy()

                else:
                    samples_by_class[label] = pd.concat(
                        [samples_by_class[label], group],
                        ignore_index=True
                    )

                # Keep maximum samples per class
                if len(samples_by_class[label]) > MAX_SAMPLES_PER_CLASS:

                    samples_by_class[label] = (
                        samples_by_class[label]
                        .sample(
                            MAX_SAMPLES_PER_CLASS,
                            random_state=RANDOM_STATE
                        )
                        .reset_index(drop=True)
                    )

    except Exception as e:
        print("File reading error:", e)


# ==========================================
# COMBINE CLASS SAMPLES
# ==========================================

if not samples_by_class:
    raise SystemExit("ERROR: No valid labeled data found.")

dataframes = list(samples_by_class.values())

df = pd.concat(
    dataframes,
    ignore_index=True
)

df = df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

print("\nTotal sampled records:", len(df))

print("\nClass distribution:")
print(df["Label"].value_counts())


# ==========================================
# CLEAN DATA
# ==========================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=["Label"]
)


# ==========================================
# FEATURES
# ==========================================

y = df["Label"].astype(str).str.strip()

X = df.drop(
    columns=["Label"],
    errors="ignore"
)

# Keep numeric features
X = X.select_dtypes(
    include=[np.number]
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan
).fillna(0)

# Remove duplicate feature columns
X = X.loc[:, ~X.columns.duplicated()]

if X.empty:
    raise SystemExit("ERROR: No numeric features found.")


# ==========================================
# REMOVE CLASSES WITH TOO FEW RECORDS
# ==========================================

class_counts = y.value_counts()

valid_classes = class_counts[
    class_counts >= 2
].index

valid_mask = y.isin(valid_classes)

X = X.loc[valid_mask].reset_index(drop=True)
y = y.loc[valid_mask].reset_index(drop=True)

print("\nClasses used for training:")
print(y.value_counts())


# ==========================================
# LABEL ENCODING
# ==========================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nAttack types:")
for i, label in enumerate(label_encoder.classes_):
    print(i, "=", label)


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

class_counts = pd.Series(y_encoded).value_counts()

number_of_classes = len(class_counts)
test_count = int(np.ceil(len(y_encoded) * 0.20))

use_stratify = (
    class_counts.min() >= 2
    and test_count >= number_of_classes
    and (len(y_encoded) - test_count) >= number_of_classes
)

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_encoded if use_stratify else None
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=150,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced",
    min_samples_leaf=1
)

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)


# ==========================================
# EVALUATION
# ==========================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n===================================")
print("MODEL TEST ACCURACY")
print("===================================")

print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)

model_data = {
    "model": model,
    "features": list(X.columns),
    "label_encoder": label_encoder
}

joblib.dump(
    model_data,
    MODEL_PATH
)

print("\n===================================")
print("MODEL SAVED SUCCESSFULLY")
print("===================================")

print("Model path:", MODEL_PATH)
print("Features:", len(X.columns))
print("Classes:", len(label_encoder.classes_))

print("\nImproved training completed!")
