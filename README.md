# 🧠 CNN Evolution: From Zero to Expert

本專案旨在深入探討卷積神經網路（CNN）的底層運算機制。我們從最原始的數學定義出發，逐行拆解程式碼邏輯，並演進至工業級的 **im2col** 矩陣運算與**自動微分（Backpropagation）**實作。

---

## 📖 目錄

* [🟢 第一關：新手版 (模擬物理運動)](#第一關新手版)
* [🟡 第二關：普通版 (記憶體區塊化)](#第二關普通版)
* [🔴 第三關：高手版 (GEMM 優化)](#第三關高手版)
* [🟣 第四關：專家版 (自動微分架構)](#第四關專家版)
* [📊 效能與正確性驗證](#效能驗證)

---

<a name="第一關新手版"></a>
## 🟢 第一關：新手版 (con_beginner)

**技術核心：滑動視窗 (Sliding Window)**

這是卷積最直覺的表現形式，完全模擬「放大鏡」在地圖上滑動的過程。

```python
def con_beginner(image, kernel):
    # 1. 解析結構：讀取影像維度 (Height, Width)
    (iH, iW) = image.shape
    (kH, kW) = kernel.shape

    # 2. 空間計算：定義輸出畫布大小 (Input - Kernel + 1)
    oH, oW = iH - kH + 1, iW - kW + 1
    
    # 3. 記憶體分配：使用 np.zeros 預先佔領空間
    output = np.zeros((oH, oW))

    # 4. 外層座標 (y, x)：控制「放大鏡左上角」的定位
    for y in range(oH):
        for x in range(oW):
            # 5. 內層座標 (i, j)：在局部區域內進行掃描
            for i in range(kH):
                for j in range(kW):
                    # 6. 計算核心：乘積加總 (Dot Product)
                    output[y, x] += image[y+i, x+j] * kernel[i, j]
    return output
```
<a name="第二關普通版"></a>

🟡 第二關：普通版 (con_normal)
技術核心：向量運算 (Vectorization)
利用 NumPy 的「切片 (Slicing)」技術，將局部區域視為一個整塊進行處理。

```python
def con_normal(image, kernel):
    (iH, iW) = image.shape
    (kH, kW) = kernel.shape
    oH, oW = iH - kH + 1, iW - kW + 1
    output = np.zeros((oH, oW))

    for y in range(oH):
        for x in range(oW):
            # 1. 切片 (Slicing)：直接在記憶體中框出一塊區塊 (View)
            window = image[y:y+kH, x:x+kW]
            # 2. 向量運算：利用 NumPy 內建的高效加權總和
            output[y, x] = np.sum(window * kernel)
    return output
```
<a name="第三關高手版"></a>

🔴 第三關：高手版 (con_master)
技術核心：GEMM 優化 (im2col)
將複雜的滑動運算徹底轉化為單一次的「大矩陣乘法」，這是現代 AI 框架（如 PyTorch）的運算標準。
```python
def con_master(image, kernel):
    (iH, iW) = image.shape
    (kH, kW) = kernel.shape
    oH, oW = iH - kH + 1, iW - kW + 1

    # 1. 空間展開 (im2col)：把所有掃描區塊全部抓出來並攤平
    # 產出一個大表格，讓電腦一次對整張表進行運算
    image_matrix = get_cols(image, kH, kW, oH, oW)

    # 2. 點乘運算 (np.dot)：通用矩陣乘法 (GEMM)
    output = np.dot(image_matrix, kernel.reshape(-1))
    
    # 3. 形狀還原：將一長條的結果重新變回 2D 地圖
    return output.reshape(oH, oW)
```
💎 核心價值：
空間換時間： 雖然會消耗更多記憶體來儲存副本，但計算速度會大幅領先迴圈版本。

<a name="第四關專家版"></a>

🟣 第四關：專家版 (ConvLayerExpert)
技術核心：自動微分與反向傳播 (Backpropagation)
這是一個完整的偵探零件，它不只會尋找特徵，還會根據正確答案來修正自己的濾鏡。
| 階段 | 行為 | 電腦科學 / 數學原理 |
| :--- | :--- | :--- |
| **Forward** | 找線索並存檔 | **儲存歷史狀態**：將輸入資料記錄於 `self.x`，作為後續反向傳播的對照基準。 |
| **Backward** | 算錯誤並檢討 | **自動微分**：利用偏微分與 **轉置矩陣 ($T$)** 運算，精確算出權重調整方向 $dW$。 |
| **Restore** | 傳遞錯誤報告 | **維度還原**：使用 `col2im` 函數將矩陣化誤差「織回」成原始圖片形狀，傳遞給前一層神經網路。 |
```python
class ConvLayerExpert:
    def forward(self, x):
        self.x = x
        # 執行高效矩陣前向傳算
        out = np.dot(col, col_W) + self.bias
        return out

    def backward(self, dout):
        # 1. 計算權重梯度 (dW)：更新濾鏡上的數值
        self.dW = np.dot(self.col.T, dout)
        # 2. 計算輸入梯度 (dx)：將誤差傳回給前一位偵探
        dx = col2im(dcol, self.x.shape, ...)
        return dx
```
<a name="效能驗證"></a>

📊 效能與正確性驗證
本專案使用 np.allclose 進行嚴謹的比對，確保所有優化版本在數學上完全等價。
```python
--- 輸出檢查 ---
新手版 Output Shape: (9, 9)
普通版 Output Shape: (9, 9)
高手版 Output Shape: (9, 9)

--- 正確性檢查 ---
普通版與新手版一致: True
高手版與新手版一致: True
```
✅ 結論： 所有優化版本的計算精度均符合標準。本專案展示了從基礎演算法到現代深度學習框架優化的演進路徑，是理解 AI 底層實作的最佳手冊。
