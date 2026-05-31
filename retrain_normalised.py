import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import os
from wrist_normalise import wrist_anchor_normalise

STEPS = {
    0: "Step 1: Palm to palm",
    1: "Step 2: Right over left",
    2: "Step 3: Fingers interlaced",
    3: "Step 4: Backs of fingers",
    4: "Step 5: Rotational thumbs",
    5: "Step 6: Fingertip rubbing",
    6: "Step 7: Wrists"
}

# Load normalised personalisation data
features = []
labels = []

for step in range(7):
    filepath = f'jetson_data_normalised/step{step}_norm.npy'
    if os.path.exists(filepath):
        data = np.load(filepath)
        features.append(data)
        labels.extend([step] * len(data))
        print(f"{STEPS[step]}: {len(data)} frames loaded")
    else:
        print(f"Missing: {filepath}")

X = np.vstack(features)
y = np.array(labels)
print(f"\nTotal frames: {len(X)}")
print(f"Feature shape: {X.shape}")

# Fit new scaler on normalised data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save new scaler
np.save('scaler_mean_norm.npy', scaler.mean_)
np.save('scaler_scale_norm.npy', scaler.scale_)
print("Scaler saved")

# Load existing model and fine-tune
booster = xgb.Booster()
booster.load_model('xgb_model_personal.json')
print("Base model loaded")

# Train on normalised data
dtrain = xgb.DMatrix(X_scaled, label=y)
params = {
    'max_depth': 8,
    'learning_rate': 0.05,
    'objective': 'multi:softprob',
    'num_class': 7,
    'eval_metric': 'mlogloss'
}

print("Training normalised model...")
model = xgb.train(
    params, dtrain,
    num_boost_round=200,
    xgb_model=booster,
    verbose_eval=50
)
model.save_model('xgb_model_norm.json')
print("\nNormalised model saved as xgb_model_norm.json")

# Quick accuracy check on training data
preds = model.predict(dtrain)
pred_labels = np.argmax(preds, axis=1)
acc = np.mean(pred_labels == y) * 100
print(f"Training accuracy: {acc:.2f}%")
print("="*50)
print("Files saved:")
print("  xgb_model_norm.json")
print("  scaler_mean_norm.npy")
print("  scaler_scale_norm.npy")
print("="*50)
