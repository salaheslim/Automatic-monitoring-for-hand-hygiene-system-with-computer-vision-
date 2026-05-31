import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import os

# Load original training data from model
# Load new Jetson personalisation data
jetson_features = []
jetson_labels = []

for step in range(7):
    filepath = f'jetson_data/step{step}_jetson.npy'
    if os.path.exists(filepath):
        data = np.load(filepath)
        jetson_features.append(data)
        jetson_labels.extend([step] * len(data))
        print(f"Step {step+1}: {len(data)} frames loaded")

X_personal = np.vstack(jetson_features)
y_personal = np.array(jetson_labels)

# Fit new scaler on personalisation data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_personal)

# Save new scaler
np.save('scaler_mean_jetson.npy', scaler.mean_)
np.save('scaler_scale_jetson.npy', scaler.scale_)

# Load existing model and fine-tune
booster = xgb.Booster()
booster.load_model('xgb_model_personal.json')

# Train new model on personalisation data
dtrain = xgb.DMatrix(X_scaled, label=y_personal)
params = {
    'max_depth': 8,
    'learning_rate': 0.05,
    'objective': 'multi:softprob',
    'num_class': 7,
    'n_estimators': 100
}
model_jetson = xgb.train(params, dtrain, num_boost_round=100,
                          xgb_model=booster)
model_jetson.save_model('xgb_model_jetson.json')
print("Jetson model saved successfully!")
