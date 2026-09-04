import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. 模擬蘋果數據集 (紅蘋果、青蘋果、網球)
# ==========================================
np.random.seed(42)

# 特徵定義：[顏色值(0紅~1綠), 甜度(0不甜~10甜)]
# 目標標籤 (Label)：1 代表「是蘋果」, 0 代表「不是蘋果(異常)」

# A. 大多數常態：紅蘋果 (1000顆) -> 偏紅、偏甜
red_apples = np.hstack([
    np.random.normal(0.1, 0.05, (1000, 1)), # 顏色接近 0 (紅)
    np.random.normal(8.0, 1.0, (1000, 1))   # 甜度高
])
y_red = np.ones(1000) # 標籤為 1

# B. 少數派但合法的同類：青蘋果 (50顆) -> 偏綠、偏酸
green_apples = np.hstack([
    np.random.normal(0.9, 0.05, (50, 1)),  # 顏色接近 1 (綠)
    np.random.normal(3.0, 1.0, (50, 1))   # 甜度低
])
y_green = np.ones(50) # 標籤依然為 1 (青蘋果也是蘋果！)

# C. 真正的異常/非蘋果：綠色網球 (30個) -> 偏綠、完全不甜
tennis_balls = np.hstack([
    np.random.normal(0.85, 0.05, (30, 1)), # 顏色接近 0.85 (綠)
    np.random.normal(0.0, 0.1, (30, 1))    # 甜度接近 0
])
y_tennis = np.zeros(30) # 標籤為 0 (不是蘋果)

# 合併所有數據
X = np.vstack([red_apples, green_apples, tennis_balls])
y = np.concatenate([y_red, y_green, y_tennis])

# 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ==========================================
# 2. 建立並訓練隨機森林模型 (加入關鍵參數)
# ==========================================
# 關鍵參數：class_weight='balanced' 
# 演算法會自動計算標籤 0 與 1 的比例，並在計算損失時，給予少數派更高的權重
rf_model = RandomForestClassifier(
    n_estimators=100, 
    class_weight='balanced', 
    random_state=42
)

rf_model.fit(X_train, y_train)

# ==========================================
# 3. 模型預測與評估
# ==========================================
y_pred = rf_model.predict(X_test)

print("--- 混淆矩陣 (Confusion Matrix) ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- 分類報告 (Classification Report) ---")
print(classification_report(y_test, y_pred, target_names=['Not Apple (0)', 'Apple (1)']))

# ==========================================
# 4. 驗證測試：青蘋果有沒有被當成蘋果？
# ==========================================
# 我們單獨拿 5 顆隨機測試集之外的青蘋果來測試
test_green_apple = np.array([[0.91, 2.8]]) # 綠色、酸
prediction = rf_model.predict(test_green_apple)
probability = rf_model.predict_proba(test_green_apple)

print("\n--- 單獨測試一棵新青蘋果 ---")
print(f"特徵: 顏色={test_green_apple[0][0]}, 甜度={test_green_apple[0][1]}")
print(f"模型預測結果: {'是蘋果 (1)' if prediction[0] == 1 else '不是蘋果 (0)'}")
print(f"預測機率: 不是蘋果={probability[0][0]:.2f}, 是蘋果={probability[0][1]:.2f}")
