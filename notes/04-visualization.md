# 04 — 視覺化與理解驗證

## Overview

Phase 4 設計了 6 個系列、共 16 張視覺化圖，逐層驗證 Phase 3 對 MPPI 的理論理解。

**資料來源：**
- `trajectory_mjwp.npz` — optimizer 輸出
- `trajectory_kinematic.npz` — IK reference

**任務：** p36-tea（bimanual, xhand）

---

## 1. Reward 收斂分析（01 系列）

### 設計動機

驗證 MPPI 的 iterative optimization 是否真的在收斂，以及不同 tick 的收斂行為是否一致。

### 觀察

| 圖表 | 內容 | 發現 |
|------|------|------|
| 01a | Reward Convergence per Tick | tick 0/5/10/16 的 reward max/mean/min 隨 iteration 上升。大部分 tick 在 1-3 個 iteration 內收斂（early stopping 生效） |
| 01b | Improvement Heatmap | mask 掉 opt_steps 之後的 iteration，improvement 集中在前幾輪，後續幾乎為零——退火 + early stopping 運作正常 |
| 01c | Cross-Tick Reward Trend | final iteration reward 跨 tick 大致穩定，無明顯崩潰——receding horizon 維持了追蹤品質 |

### 理論驗證

符合 Phase 3 對 MPPI 收斂機制的理解：
> 退火降 noise → weighted average 精化 → early stopping 截斷

但也確認了 **「收斂 ≠ 最優」**——只是 improvement 低於閾值就停了。

---

## 2. Tracking Reward 分解（02 系列）

### 設計動機

SPIDER 的 cost function 有 qpos（位置）和 qvel（速度）兩項。想知道哪個主導了 optimizer 行為。

### 觀察

| 圖表 | 內容 | 發現 |
|------|------|------|
| 02a | qpos / qvel Reward over Ticks | qpos reward 絕對值遠大於 qvel，量級差 2-3 個 order of magnitude |
| 02b | Reward Component Ratio | qpos 佔 cost >99.7%，qvel 幾乎可忽略 |
| 02c | Optimizer Effort | opt_steps 高的 tick 很少，大部分 1-2 步就 early stop |

### 理論修正

> ⚠️ 原本以為位置和速度 tracking 同等重要。視覺化後發現 qvel 的權重設定讓它幾乎不影響優化方向。

這是一個設計選擇：SPIDER 優先保證 **「手到對的位置」**，而非「手以對的速度到達」。

---

## 3. 軌跡 vs Reference（03 系列）

### 設計動機

直接比較 MPPI 輸出的 sim trajectory 和 kinematic IK reference，驗證物理軌跡對 reference 的追蹤品質。

### 觀察

| 圖表 | 內容 | 發現 |
|------|------|------|
| 03a | Joint Timeseries | 8 個代表性 joint 的 qpos vs reference。大部分 DOF 追蹤良好，偏差集中在特定 finger joints，與 contact transition 位置相關 |
| 03b | Tick Boundary Discontinuity | tick 邊界 \|Δqpos\| heatmap — mean 0.004, max 0.068。大部分 DOF 的 tick boundary 非常平滑（warm start 有效）。Max 出現在 finger joints，可能是 contact transition 造成的 replanning 跳變 |
| 03c | Per-Tick qpos Range | wrist DOF 變化大（大範圍移動），finger DOF 變化小（精細調整） |

### 理論驗證

Receding horizon 的 shift + warm start 確實維持了 tick 間連續性。不連續性集中在 contact transition 附近，符合 Phase 3「contact boundary 造成映射不連續」的預測。

---

## 4. 控制信號分析（04 系列）

### 設計動機

軌跡好 ≠ 控制信號好。想知道 actuator ctrl 是否平滑，以及控制努力和優化難度的關係。

### 觀察

| 圖表 | 內容 | 發現 |
|------|------|------|
| 04a | Ctrl Timeseries | 8 個 actuator 的控制信號大致平滑，tick 邊界有微小跳變（對應 03b） |
| 04b | Ctrl Discontinuity Heatmap | tick 邊界 \|Δctrl\| — mean 0.012, max 0.089。比 qpos 跳變更明顯但仍可接受 |
| 04c | Ctrl Effort vs opt_steps | ctrl_effort 和 opt_steps 相關性弱（r = 0.168）；ctrl_rate 和 opt_steps 弱負相關（r = −0.305） |

### 理論修正

> ⚠️ 原本預期「optimizer 花越多步的 tick，控制信號應該越大/變化越快」。但數據顯示幾乎無關。

**解釋：** opt_steps 高的原因是 reward landscape 複雜（contact），不是因為需要大控制量。控制信號大小由 **任務幾何** 決定，而非優化難度。

---

## 5. 3D 空間軌跡（05 系列）

### 設計動機

在 joint space 看到的都是抽象數字。想在 Cartesian space 看 fingertip 的實際運動路徑，特別是 receding horizon 的 plan vs execute 差異。

### 觀察

| 圖表 | 內容 | 發現 |
|------|------|------|
| 05a | Receding Horizon 視覺化 | 3 個 representative sites（L-wrist, L-index, R-index），兩個視角。每個 tick 規劃 160 步（虛線），只執行前 40 步（實線, 25%）。**虛線尾端散開**——離當前狀態越遠的規劃越不可靠 |
| 05b | Per-Tick Site Displacement | Site 1（L-thumb）幾乎不動（位移 ~0），wrist sites 位移最大。左右手活動量不對稱 |
| 05c | Bimanual Paths | 左右手平均軌跡 + gap detection。Inter-tick gaps ~1-2mm，幾乎連續 |

### 理論驗證

Receding horizon 的核心假設（「只信近期」）在 3D 空間中得到直觀驗證：
- **遠端規劃散開** — 虛線發散
- **近端執行緊密** — 實線收斂
- **Tick 間空間連續** — 與 03b 的 joint-space 結果一致

---

## 6. 跨 Rollout / 跨任務比較（06 系列）

### 設計動機

單一成功 rollout 看不出 SPIDER 的弱點。需要跨 rollout（同任務不同隨機種子）和跨任務（不同操作難度）的比較。

### 觀察

| 圖表 | 內容 | 發現 |
|------|------|------|
| 06a | Reward Across Rollouts | p36-tea: rollout 0 在所有 17 個 tick 都是 outlier（reward 顯著低於 mean − 1σ）；p44-dog / p52-instrument 一致性高 |
| 06b | Optimizer Effort Profile | p36-tea peak 在 tick 9（contact transition）；p44-dog peak 在 tick 0（開局接觸物體）；p52-instrument peak 在 tick 4 |
| 06c | Cross-Task Difficulty | 難度排名：**p36-tea (6.8) > p52-instrument (4.5) > p44-dog (1.7)**（mean opt_steps）。p36-tea 涉及複雜 bimanual 協調 + contact transition |
| 06d | Reward Variance | Reward variance 的 peak 位置可作為 **contact instability indicator**。p36-tea variance 遠高於其他兩個任務 |

### 理論驗證

Phase 3 預測的「contact boundary 不連續性」在跨 rollout 比較中得到量化驗證：

1. **Contact transition tick 的 opt_steps spike** — 符合「feasible set 收縮」的預測
2. **p36-tea rollout 0 整體 outlier** — 初始 noise seed 可能落入 homotopy path 斷裂的區域
3. **任務越複雜，MPPI 穩定性越差** — 符合「O(2^{nm}) contact modes vs 1024 samples」的覆蓋率問題

---

## Key Takeaways

### 核心認知

1. **Receding Horizon 的直觀意義：** 規劃 1.6s 只執行 0.4s 不是浪費——用遠端計劃提供梯度方向，近端執行提供精確控制。3D 圖中虛線散開、實線緊密，完美體現設計理念。

2. **不同 DOF 的行為差異是合理的：** 同時優化 ≠ 同等重要。Wrist 負責大範圍移動，fingers 負責精細接觸。qpos >99.7% 的 cost share 說明策略是 **「先到位，再談力」**。

3. **成功案例的表象會騙人：** 單一成功 rollout 看起來一切正常。只有跨 rollout 比較才能暴露 MPPI 在 contact transition 附近的脆弱性。

4. **Ctrl effort ≠ 優化難度：** 控制信號大小由任務幾何決定，opt_steps 多寡由 reward landscape 複雜度決定。弱相關是合理的——回答的是不同問題。

### 四個關鍵診斷指標

| 指標 | 衡量什麼 |
|------|----------|
| `opt_steps` | 優化收斂速度（越高 = 越難） |
| Reward 水準 | 追蹤品質（越負 = 越差） |
| Tick 邊界連續性 | Receding horizon 銜接品質 |
| 跨 rollout reward variance | 穩定性 / contact instability |

---

## Completion Checklist

- [x] 至少一個自己寫的視覺化，能對應到論文的某個 figure（06 系列的跨任務比較 → 論文 Table 1 的 success rate 排名）
- [x] 如果結果不同，能解釋為什麼（我們看的是 optimizer effort 而非 success rate，但難度排名一致）
