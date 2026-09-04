import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# ==========================================
# 1. 準備您的數據 (此處為模擬 UWB/通訊信號特徵)
# ==========================================
# 特徵矩陣 X: [Kurtosis, K-factor, Variance]
# 標籤 y: 0 = LOS (視距常態), 1 = NLOS (非視距擋牆)

np.random.seed(42)
n_samples = 2000

# 模擬 LOS 信號：高峰度、高K-factor、低方差
los_features = np.hstack([
    np.random.normal(15.0, 2.0, (1200, 1)),  # Kurtosis 高
    np.random.normal(8.0, 1.5, (1200, 1)),   # K-factor 高
    np.random.normal(0.5, 0.2, (1200, 1))    # Variance 低
])
y_los = np.zeros(1200)

# 模擬 NLOS 信號：低峰度、低K-factor、高方差 (多徑效應)
nlos_features = np.hstack([
    np.random.normal(3.0, 1.0, (800, 1)),    # Kurtosis 低
    np.random.normal(0.5, 0.3, (800, 1)),    # K-factor 低
    np.random.normal(4.5, 1.2, (800, 1))     # Variance 高
])
y_nlos = np.ones(800)

# 合併數據集
X = np.vstack([los_features, nlos_features])
y = np.concatenate([y_los, y_nlos])

# 切分訓練與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==========================================
# 2. 建立隨機森林 (N)LOS 檢測器
# ==========================================
# 使用 class_weight='balanced' 自動調節環境中兩者的比例不平衡
nlos_detector = RandomForestClassifier(
    n_estimators=100, 
    max_depth=6, 
    class_weight='balanced',
    random_state=42
)

nlos_detector.fit(X_train, y_train)

# ==========================================
# 3. 預測與輸出 NLOS 軟判決（Soft Decision 機率）
# ==========================================
# 提取預測為 NLOS (1) 的機率，完美取代您原先手動算的 nlos_prob 
nlos_prob_test = nlos_detector.predict_proba(X_test)[:, 1]
y_pred = nlos_detector.predict(X_test)

print("--- (N)LOS 檢測器評估報告 ---")
print(classification_report(y_test, y_pred, target_names=['LOS (0)', 'NLOS (1)']))
print(f"ROC AUC 檢測效能分數: {roc_auc_score(y_test, nlos_prob_test):.4f}")

# ==========================================
# 4. 查看各個特徵在演算法眼中的真實權重
# ==========================================
print("\n--- 演算法自動學習到的真實特徵重要性 (取代手動定權重) ---")
features_names = ['Kurtosis', 'K-factor', 'Variance']
for name, importance in zip(features_names, nlos_detector.feature_importances_):
    print(f"{name} 的科學貢獻權重: {importance:.4f}")
