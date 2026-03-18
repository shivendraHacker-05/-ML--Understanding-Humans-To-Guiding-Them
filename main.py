import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.sparse import hstack

# ======================
# LOAD TRAIN DATA
# ======================
train_df = pd.read_excel("Sample_arvyax_reflective_dataset.xlsx")

# ======================
# CLEANING
# ======================
train_df.fillna({
    "sleep_hours": train_df["sleep_hours"].mean(),
    "energy_level": train_df["energy_level"].median(),
    "stress_level": train_df["stress_level"].median()
}, inplace=True)

train_df["journal_text"] = train_df["journal_text"].astype(str)

# ======================
# TEXT FEATURES
# ======================
tfidf = TfidfVectorizer(max_features=300)
X_text_train = tfidf.fit_transform(train_df["journal_text"])

# ======================
# METADATA
# ======================
meta_cols = ["sleep_hours", "energy_level", "stress_level", "duration_min"]
X_meta_train = train_df[meta_cols]

# Combine
X_train = hstack([X_text_train, X_meta_train])

# Targets
y_state = train_df["emotional_state"]
y_intensity = train_df["intensity"]

# ======================
# TRAIN MODELS
# ======================
clf = RandomForestClassifier()
reg = RandomForestRegressor()

clf.fit(X_train, y_state)
reg.fit(X_train, y_intensity)

# ======================
# LOAD TEST DATA
# ======================
test_df = pd.read_excel("arvyax_test_inputs_120.xlsx")

# Fill missing values (same logic)
test_df.fillna({
    "sleep_hours": train_df["sleep_hours"].mean(),
    "energy_level": train_df["energy_level"].median(),
    "stress_level": train_df["stress_level"].median()
}, inplace=True)

test_df["journal_text"] = test_df["journal_text"].astype(str)

# ======================
# TRANSFORM TEST DATA
# ======================
X_text_test = tfidf.transform(test_df["journal_text"])
X_meta_test = test_df[meta_cols]

X_test = hstack([X_text_test, X_meta_test])

# ======================
# PREDICT
# ======================
pred_state = clf.predict(X_test)
pred_intensity = reg.predict(X_test)

# ======================
# CONFIDENCE
# ======================
probs = clf.predict_proba(X_test)
confidence = probs.max(axis=1)

uncertain_flag = (confidence < 0.6).astype(int)

# ======================
# DECISION ENGINE
# ======================
def decide_action(state, intensity, stress, energy, time):
    if stress > 7 and energy < 4:
        return "box_breathing", "now"
    if energy > 7:
        return "deep_work", "now"
    if energy < 3:
        return "rest", "within_15_min"
    if time == "night":
        return "sleep", "tonight"
    return "light_planning", "later_today"

actions = []
timings = []

for i in range(len(test_df)):
    a, t = decide_action(
        pred_state[i],
        pred_intensity[i],
        test_df["stress_level"][i],
        test_df["energy_level"][i],
        test_df["time_of_day"][i]
    )
    actions.append(a)
    timings.append(t)

# ======================
# SAVE OUTPUT
# ======================
output = pd.DataFrame({
    "id": test_df["id"],
    "predicted_state": pred_state,
    "predicted_intensity": pred_intensity,
    "confidence": confidence,
    "uncertain_flag": uncertain_flag,
    "what_to_do": actions,
    "when_to_do": timings
})

output.to_csv("predictions2.csv", index=False)

print("✅ DONE — predictions2.csv generated")