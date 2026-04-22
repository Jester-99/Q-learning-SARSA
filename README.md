# Q-Learning vs SARSA：Cliff Walking 強化學習比較

> 本作業實作並比較兩種經典強化學習演算法——**Q-Learning（離策略）** 與 **SARSA（同策略）**，  
> 在相同的 Cliff Walking 環境與參數設定下，分析學習行為、收斂特性與最終策略差異。

---

## 目錄

- [環境描述](#環境描述)
- [程式設計邏輯](#程式設計邏輯)
  - [專案結構](#專案結構)
  - [環境實作](#環境實作)
  - [演算法實作](#演算法實作)
  - [實驗流程](#實驗流程)
- [實驗結果](#實驗結果)
  - [學習曲線](#學習曲線)
  - [策略視覺化](#策略視覺化)
  - [穩定性分析](#穩定性分析)
  - [整合儀表板](#整合儀表板)
- [統計比較](#統計比較)
- [結論](#結論)
- [執行方式](#執行方式)

---

## 環境描述

本實驗採用 **Cliff Walking** 環境——一個 **4 × 12** 的格子世界：

```
 Col:  0    1    2    3    4    5    6    7    8    9   10   11
Row 0 [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]
Row 1 [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]
Row 2 [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]
Row 3 [S]  [C]  [C]  [C]  [C]  [C]  [C]  [C]  [C]  [C]  [C]  [G]
       Start          Cliff (懸崖)                          Goal
```

| 元素 | 位置 | 獎勵 |
|------|------|------|
| 起點 Start | `(3, 0)` | — |
| 終點 Goal  | `(3, 11)` | −1（結束回合） |
| 懸崖 Cliff | `(3, 1)` ～ `(3, 10)` | **−100**，回到起點 |
| 一般格子   | 其餘格子 | **−1** |

---

## 程式設計邏輯

### 專案結構

```
rl_cliff_walking/
├── cliff_walking.py        # 主程式（環境 + 演算法 + 視覺化）
├── README.md               # 說明文件
├── learning_curves.png     # 學習曲線圖
├── policy_visualization.png # 策略視覺化圖
├── stability_analysis.png  # 穩定性分析圖
└── summary_dashboard.png   # 整合儀表板
```

---

### 環境實作

#### `class CliffWalking`

核心環境類別，封裝所有環境邏輯：

```python
class CliffWalking:
    ROWS = 4
    COLS = 12
    START = (3, 0)
    GOAL  = (3, 11)
    ACTIONS = [(-1,0), (1,0), (0,-1), (0,1)]  # 上、下、左、右
```

**關鍵方法：**

| 方法 | 功能 |
|------|------|
| `reset()` | 將 Agent 重置至起點，回傳初始狀態 |
| `step(action)` | 執行動作，回傳 `(next_state, reward, done)` |
| `is_cliff(state)` | 判斷是否為懸崖格（`row==3` 且 `1<=col<=10`） |
| `is_goal(state)` | 判斷是否到達終點 |
| `state_to_idx(state)` | 將 `(row, col)` 轉換為線性索引，用於 Q 表 |

**`step()` 流程：**

```
執行動作 → 計算新座標（邊界 clip）
      ↓
  是否為懸崖？ → Yes → reward = -100, next_state = START
      ↓ No
  是否為終點？ → Yes → reward = -1, done = True
      ↓ No
  reward = -1, done = False
```

---

### 演算法實作

#### ε-greedy 策略

```python
def epsilon_greedy(Q, state_idx, epsilon, n_actions):
    if random() < epsilon:
        return randint(0, n_actions)   # 隨機探索
    else:
        return argmax(Q[state_idx])    # 貪婪選擇
```

以機率 ε 進行**隨機探索**，以 1-ε 機率選擇**當前最佳動作**。

---

#### Q-Learning（Off-policy）

```python
def q_learning(env, episodes, alpha, gamma, epsilon):
    Q = zeros(n_states, n_actions)        # 初始化 Q 表
    for each episode:
        state = env.reset()
        while not done:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done = env.step(action)

            # Off-policy 更新：使用下一狀態的最大 Q 值
            best_next = max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

            state = next_state
```

**更新公式：**

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$$

- **目標值** 使用下一狀態「最優動作」的 Q 值
- 更新策略（Greedy）≠ 行為策略（ε-greedy）→ **離策略**

---

#### SARSA（On-policy）

```python
def sarsa(env, episodes, alpha, gamma, epsilon):
    Q = zeros(n_states, n_actions)        # 初始化 Q 表
    for each episode:
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)    # 先選好第一個動作
        while not done:
            next_state, reward, done = env.step(action)

            # On-policy 更新：使用實際選擇的下一個動作
            next_action = epsilon_greedy(Q, next_state, epsilon)
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

            state, action = next_state, next_action
```

**更新公式：**

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma Q(s',a') - Q(s,a) \right]$$

- **目標值** 使用實際採取的下一個動作 a' 的 Q 值
- 更新策略 = 行為策略 = ε-greedy → **同策略**

---

#### 兩者差異對照

| 特性 | Q-Learning | SARSA |
|------|-----------|-------|
| 策略類型 | Off-policy（離策略） | On-policy（同策略） |
| 目標 Q 值 | `max Q(s', *)` | `Q(s', a')` 其中 a' 由 ε-greedy 選出 |
| 更新反映探索風險？ | ❌ 否（假設總是貪婪） | ✅ 是（反映實際探索行為） |
| 收斂至 | 最優策略 π* | ε-greedy 下的最優策略 |

---

### 實驗流程

```
run_multiple(algo, env, n_runs=50)
      │
      ├─ 執行 50 次獨立實驗
      │       每次：初始化環境 → 訓練 500 回合 → 記錄每回合 total reward
      │
      ├─ all_rewards: shape (50, 500)
      │       mean(axis=0) → 平均學習曲線
      │       std(axis=0)  → 標準差（用於信賴區間）
      │
      ├─ Q_avg = mean(all_Q, axis=0) → 50 次平均 Q 表（更具代表性）
      │
      └─ 回傳 mean, std, Q_avg, all_rewards
```

使用 50 次重複實驗取平均，確保統計可靠性，消除隨機性影響。

---

## 實驗結果

### 學習曲線

> 每回合累積獎勵（Total Reward），含 95% 信賴區間與收斂標記。

![學習曲線](learning_curves.png)

**觀察：**
- **早期（0～50 回合）**：兩者均從極低報酬（多次掉落懸崖）快速爬升
- **SARSA（青色）**：約在第 **206 回合**穩定超過 −30，收斂較快且波動小
- **Q-Learning（紅色）**：報酬持續在 −45 ～ −30 之間震盪，難以穩定超越 −30 門檻
- 平滑曲線圖中的**垂直虛線**標示各演算法的收斂回合，**水平虛線**為 −30 門檻

---

### 策略視覺化

> 訓練完成後，依據平均 Q 表執行貪婪策略所學到的路徑。

![策略視覺化](policy_visualization.png)

**路徑分析：**

| | Q-Learning | SARSA |
|-|-----------|-------|
| **路徑風格** | 沿懸崖邊緣（Row 3）直行 | 繞道上方（Row 1 或 2） |
| **路徑長度** | 較短（理論最優 11 步） | 較長（較安全但需多步） |
| **風險評估** | ε-greedy 探索時易掉入懸崖 | 保守策略，遠離懸崖 |

**原因解析：**
- Q-Learning 的 Off-policy 更新假設「下一步一定貪婪」，因此不把探索失誤計入 Q 值，導致它「不知道」懸崖邊緣的實際危險
- SARSA 的 On-policy 更新直接反映 ε-greedy 探索的代價，ε=0.1 意味著 10% 機率隨機走到懸崖，這個 −100 懲罰被納入 Q 值，促使策略遠離懸崖

---

### 穩定性分析

> 比較兩種演算法在訓練過程中的波動程度。

![穩定性分析](stability_analysis.png)

**左圖：滾動標準差（window=20）**  
反映學習曲線逐回合的局部波動大小：
- Q-Learning 的滾動標準差**持續偏高**，代表學習過程不穩定
- SARSA 的標準差在約 100 回合後**快速下降**並保持低位

**右圖：後半段跨實驗報酬分布**  
取 50 次獨立實驗中每個 run 後半段（250～500 回合）的平均報酬：
- SARSA 的箱型圖**中位數更高、四分位距更窄**，代表實驗間結果一致性高
- Q-Learning 的分布更分散，部分 run 表現優異，但整體不穩定

---

### 整合儀表板

> 深色主題的完整分析總覽。

![整合儀表板](summary_dashboard.png)

---

## 統計比較

| 指標 | Q-Learning | SARSA |
|------|-----------|-------|
| 全程平均報酬 | −72.34 | **−51.37** |
| 全程報酬標準差 | 74.47 | **39.31** |
| 後半段平均報酬 | −49.32 | **−23.65** |
| 後半段報酬標準差 | 66.74 | **21.63** |
| 最高單回合報酬 | −25.34 | **−17.92** |
| 首次收斂回合（>−30） | ~500（末期勉強達標） | **~206** |

---

## 結論

| 面向 | 結論 |
|------|------|
| **收斂速度** | SARSA 更快（~206 回合），Q-Learning 在此設定下難穩定收斂 |
| **訓練穩定性** | SARSA 標準差約 Q-Learning 的 **1/3**，顯著更穩定 |
| **最終策略** | Q-Learning：懸崖邊最短路（理論最優）；SARSA：安全繞行路徑 |
| **探索影響** | ε=0.1 的隨機性對 SARSA 是訓練信號，對 Q-Learning 是噪音 |
| **適用情境** | 學習期安全優先 → **SARSA**；純貪婪部署 → **Q-Learning** |

> **核心結論**：在 ε-greedy 探索下，SARSA 的同策略更新機制使其能感知並規避探索風險，在 Cliff Walking 這類邊界敏感環境中表現更穩定。若探索率 ε→0，兩者將收斂至相同的最優策略。

---

## 執行方式

```bash
# 安裝相依套件
pip install numpy matplotlib

# 執行主程式（約 30～60 秒）
python cliff_walking.py
```

執行後自動生成 4 張分析圖，並在終端機輸出統計報告。

**超參數設定（可在 `cliff_walking.py` 第 609 行調整）：**

```python
EPISODES  = 500    # 訓練回合數
ALPHA     = 0.1    # 學習率
GAMMA     = 0.9    # 折扣因子
EPSILON   = 0.1    # 探索率
N_RUNS    = 50     # 重複實驗次數
```

---

## 參考資料

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
  - Chapter 6：Temporal-Difference Learning
  - Example 6.6：Cliff Walking
