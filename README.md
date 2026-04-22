# 強化學習作業報告：Q-Learning vs SARSA

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-11557c)
![License](https://img.shields.io/badge/License-MIT-green)

**比較 Q-Learning（Off-policy）與 SARSA（On-policy）在 Cliff Walking 環境的學習行為**

</div>

---

## 📋 目錄

1. [作業目的](#一作業目的)
2. [環境描述](#二環境描述)
3. [演算法理論](#三演算法理論)
4. [程式設計邏輯](#四程式設計邏輯)
5. [實驗設定](#五實驗設定)
6. [實驗結果](#六實驗結果)
7. [結果分析與討論](#七結果分析與討論)
8. [理論比較](#八理論比較)
9. [結論](#九結論)
10. [執行方式](#十執行方式)
11. [參考資料](#參考資料)

---

## 一、作業目的

本作業旨在實作並比較兩種經典強化學習演算法——**Q-learning** 與 **SARSA**，透過相同環境與參數設定，分析其：

- **學習行為**：每回合累積獎勵的變化趨勢
- **收斂特性**：達到穩定策略所需的回合數
- **最終策略差異**：兩種演算法學到的行動路徑比較

---

## 二、環境描述

### 2.1 Cliff Walking 格子世界

本實驗採用 **4 × 12 Cliff Walking** 環境（Sutton & Barto, 2018, Example 6.6）：

```
Col:   0    1    2    3    4    5    6    7    8    9   10   11
      ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
Row 0 │    │    │    │    │    │    │    │    │    │    │    │    │
      ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
Row 1 │    │    │    │    │    │    │    │    │    │    │    │    │
      ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
Row 2 │    │    │    │    │    │    │    │    │    │    │    │    │
      ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
Row 3 │ S  │ ██ │ ██ │ ██ │ ██ │ ██ │ ██ │ ██ │ ██ │ ██ │ ██ │ G  │
      └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
       起點                    懸 崖 (Cliff)                    終點
```

### 2.2 環境規格

| 元素 | 位置 | 說明 |
|------|------|------|
| **起點 Start** | `(3, 0)` | Agent 每回合的起始位置 |
| **終點 Goal** | `(3, 11)` | 到達即結束回合 |
| **懸崖 Cliff** | `(3, 1)` ～ `(3, 10)` | 踏入 → 獎勵 −100，強制回到起點 |
| **一般格子** | 其餘 34 個格子 | 每步獎勵 −1 |

### 2.3 問題設定

| 項目 | 規格 |
|------|------|
| **狀態空間** | 48 個格子位置（4 × 12） |
| **動作空間** | 4 個（上↑、下↓、左←、右→） |
| **邊界處理** | 撞牆則停在原地（`np.clip`） |
| **回合終止** | 到達終點 Goal |

---

## 三、演算法理論

### 3.1 Q-Learning（離策略，Off-Policy）

Q-Learning 是一種**時序差分（Temporal Difference, TD）**學習方法，屬於**離策略**演算法。

**更新公式：**

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \Big[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \Big]$$

其中：
- $\alpha$：學習率（Learning Rate）
- $\gamma$：折扣因子（Discount Factor）
- $r_{t+1}$：執行動作後的即時獎勵
- $\max_{a'} Q(s_{t+1}, a')$：下一狀態所有動作中的**最大 Q 值**

**核心特性：**
- 目標值使用「下一狀態的最優動作」，**無論實際採取何種動作**
- 更新策略（Greedy）與行為策略（ε-greedy）**不同** → **Off-policy**
- 理論上收斂至最優策略 $\pi^*$

---

### 3.2 SARSA（同策略，On-Policy）

SARSA 同樣是 TD 方法，但屬於**同策略**演算法。名稱來自其更新所用的五元組：$(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$。

**更新公式：**

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \Big[ r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \Big]$$

其中 $a_{t+1}$ 是由**同一個 ε-greedy 策略**在 $s_{t+1}$ 選出的動作。

**核心特性：**
- 目標值使用「實際採取的下一個動作」的 Q 值
- 更新策略 = 行為策略 = ε-greedy → **On-policy**
- 收斂至 ε-greedy 策略下的最優策略

---

### 3.3 Off-Policy vs On-Policy 本質差異

```
ε-greedy 探索時，有 10% 機率隨機走向懸崖...

Q-Learning 的反應：
  Q(s,a) 更新時用 max Q(s') → 假設下一步一定貪婪
  → Q 值「不感知」探索失誤的代價
  → 策略在懸崖邊緣顯得過於樂觀

SARSA 的反應：
  Q(s,a) 更新時用實際選出的 Q(s', a')
  → 若 a' 因隨機探索走入懸崖，-100 懲罰被反映至 Q 值
  → 策略學會「保持距離、遠離懸崖」
```

---

## 四、程式設計邏輯

### 4.1 程式架構

```
cliff_walking.py
│
├── class CliffWalking          # 環境定義
│   ├── reset()                 # 初始化狀態
│   ├── step(action)            # 執行動作，回傳 (next_state, reward, done)
│   ├── is_cliff(state)         # 判斷是否為懸崖
│   └── state_to_idx(state)     # 狀態 → Q表索引
│
├── epsilon_greedy()            # ε-greedy 動作選擇
│
├── q_learning()                # Q-Learning 演算法
├── sarsa()                     # SARSA 演算法
│
├── run_multiple()              # 多次重複實驗（取統計平均）
├── extract_greedy_path()       # 提取貪婪策略路徑
│
└── Visualization
    ├── plot_learning_curves()  # 學習曲線（含收斂標記）
    ├── plot_policy_and_path()  # 策略箭頭 + 最優路徑
    ├── plot_stability_analysis() # 穩定性分析
    └── plot_summary_dashboard()  # 整合儀表板
```

### 4.2 環境類別：`CliffWalking`

```python
class CliffWalking:
    ROWS = 4;  COLS = 12
    START = (3, 0);  GOAL = (3, 11)
    ACTIONS = [(-1,0), (1,0), (0,-1), (0,1)]  # 上、下、左、右
```

**`step(action)` 執行流程：**

```
輸入：action ∈ {0,1,2,3}
  │
  ├─ 計算新座標（邊界 clip 防止越界）
  │     nr = clip(r + dr, 0, ROWS-1)
  │     nc = clip(c + dc, 0, COLS-1)
  │
  ├─ is_cliff(next_state)?
  │     ├─ YES → reward = -100, next_state = START, done = False
  │     └─ NO  →
  │           is_goal(next_state)?
  │             ├─ YES → reward = -1, done = True
  │             └─ NO  → reward = -1, done = False
  │
  └─ 更新 self.state，回傳 (next_state, reward, done)
```

### 4.3 Q-Learning 實作

```python
def q_learning(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    Q = np.zeros((env.n_states, env.n_actions))   # 初始化 Q 表（全零）

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            # 1. ε-greedy 選擇動作
            action = epsilon_greedy(Q, state_idx, epsilon, n_actions)

            # 2. 與環境互動
            next_state, reward, done = env.step(action)

            # 3. Off-policy 更新（使用下一狀態最大 Q 值）
            best_next = np.max(Q[next_state_idx])
            Q[s_idx, action] += alpha * (reward + gamma * best_next - Q[s_idx, action])

            total_reward += reward
            if done: break

    return Q, rewards_per_episode
```

### 4.4 SARSA 實作

```python
def sarsa(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    Q = np.zeros((env.n_states, env.n_actions))

    for ep in range(episodes):
        state = env.reset()
        # 關鍵：在回合開始時就先選好第一個動作
        action = epsilon_greedy(Q, state_idx, epsilon, n_actions)

        while True:
            next_state, reward, done = env.step(action)

            # 關鍵：選出實際的下一個動作（On-policy）
            next_action = epsilon_greedy(Q, next_state_idx, epsilon, n_actions)

            # On-policy 更新（使用實際選擇的 next_action）
            Q[s_idx, action] += alpha * (reward + gamma * Q[ns_idx, next_action] - Q[s_idx, action])

            state, action = next_state, next_action
            if done: break

    return Q, rewards_per_episode
```

### 4.5 多次實驗設計（統計可靠性）

```python
def run_multiple(algo_fn, env_class, n_runs=50, ...):
    all_rewards = []   # 收集每次 run 的獎勵序列
    all_Q = []         # 收集每次的 Q 表

    for _ in range(n_runs):
        env = env_class()
        Q, rewards = algo_fn(env, ...)
        all_rewards.append(rewards)
        all_Q.append(Q)

    all_rewards = np.array(all_rewards)        # shape: (50, 500)
    Q_avg = np.mean(np.array(all_Q), axis=0)  # 平均 Q 表

    return (
        all_rewards.mean(axis=0),   # 每回合平均報酬
        all_rewards.std(axis=0),    # 每回合標準差（用於信賴區間）
        Q_avg,                      # 平均 Q 表（用於策略視覺化）
        all_rewards                 # 完整矩陣（用於穩定性分析）
    )
```

使用 **50 次獨立重複實驗** 的設計，消除隨機種子的影響，確保統計結論的可靠性。

---

## 五、實驗設定

### 5.1 超參數

| 超參數 | 符號 | 值 | 說明 |
|--------|------|----|------|
| 學習率 | α | **0.1** | 每次更新的步長大小 |
| 折扣因子 | γ | **0.9** | 未來獎勵的重視程度 |
| 探索率 | ε | **0.1** | ε-greedy 中隨機探索的機率 |
| 訓練回合數 | — | **500** | 每次實驗的訓練總回合 |
| 重複實驗次數 | — | **50** | 用於統計平均 |
| Q 表初始值 | — | 0 | 樂觀初始化（Optimistic Initialization）使用 0 |

### 5.2 實驗流程圖

```
 Q-Learning              SARSA
     │                     │
     ▼                     ▼
run_multiple × 50    run_multiple × 50
     │                     │
     ├─ mean curve         ├─ mean curve
     ├─ std curve          ├─ std curve
     ├─ avg Q-table        ├─ avg Q-table
     └─ all_rewards(50×500)└─ all_rewards(50×500)
               │                 │
               ▼                 ▼
         ┌─────────────────────────┐
         │     統計分析 & 視覺化    │
         │  1. 學習曲線 + 收斂標記  │
         │  2. 策略箭頭 + 最優路徑  │
         │  3. 穩定性分析           │
         │  4. 整合儀表板           │
         └─────────────────────────┘
```

---

## 六、實驗結果

### 6.1 學習曲線

> 每回合累積獎勵隨訓練回合數的變化（50 次實驗平均，陰影為 95% 信賴區間）

![學習曲線](learning_curves.png)

**說明：**
- **左圖**：原始曲線，可看到每回合的實際波動
- **右圖**：平滑曲線（移動平均 window=10），黑色虛線為收斂門檻（−30），垂直虛線標示各演算法首次穩定超越門檻的回合

---

### 6.2 策略視覺化

> 訓練完成後，依照 50 次實驗的平均 Q 表，以貪婪策略執行所學到的最優路徑

![策略視覺化](policy_visualization.png)

**說明：**
- 箭頭代表每個格子的貪婪動作方向
- 彩色路徑線顯示從起點到終點的實際軌跡
- 可觀察到兩種演算法的路徑策略顯著不同

---

### 6.3 穩定性分析

> 比較兩種演算法在訓練過程中的波動程度與跨實驗一致性

![穩定性分析](stability_analysis.png)

**說明：**
- **左圖（滾動標準差）**：以 window=20 計算局部標準差，反映學習曲線的逐回合波動大小
- **右圖（箱型圖）**：取 50 次獨立實驗中，每個 run 後半段（250～500 回合）的平均報酬進行分布比較

---

### 6.4 整合儀表板

> 整合學習曲線、穩定性分布與策略網格的完整分析視圖

![整合儀表板](summary_dashboard.png)

---

## 七、結果分析與討論

### 7.1 學習表現

| 指標 | Q-Learning | SARSA |
|------|-----------|-------|
| 全程平均報酬 | −72.34 | **−51.37** |
| 後半段平均報酬 | −49.32 | **−23.65** |
| 首次超越 −30 門檻 | ~500 回合（末期勉強） | **~206 回合** |

**SARSA 收斂更快**的原因：On-policy 更新直接反映探索策略的代價，Q 值能更快反映實際環境動態，使策略在更早的回合趨於穩定。

---

### 7.2 策略行為

| | Q-Learning | SARSA |
|-|-----------|-------|
| **路徑偏好** | 沿懸崖邊緣（Row 3）直行最短路 | 繞道 Row 1/2 的安全路徑 |
| **策略風格** | 冒險型（Adventurous） | 保守型（Conservative） |
| **理論路徑長度** | 11 步（最短） | 13～17 步（較長但安全） |

**原因分析：**

```
Q-Learning 的盲點：
  更新用 max Q(s') → 假設「下一步必定貪婪」
  → 沿懸崖邊走的 Q 值被高估（沒考慮探索時掉落的代價）
  → 最終策略選擇「風險最高但看似最優」的懸崖邊路線

SARSA 的謹慎：
  更新用實際的 Q(s', a') → 10% 機率 a' 是隨機動作
  → 懸崖邊的狀態偶爾因隨機探索獲得 -100
  → Q 值被拉低，策略學會主動遠離懸崖
```

---

### 7.3 穩定性比較

| 指標 | Q-Learning | SARSA |
|------|-----------|-------|
| 全程報酬標準差 | 74.47 | **39.31** |
| 後半段報酬標準差 | 66.74 | **21.63** |
| 滾動標準差（後半段） | 持續偏高 | **快速下降後穩定** |
| 跨 50 次 run 四分位距 | 寬 | **窄** |

**SARSA 顯著更穩定**：其策略遠離懸崖，即使在探索階段偶爾隨機行動，也不易掉入懸崖獲得 −100 的極大懲罰，因此報酬波動較小。

---

### 7.4 探索（Exploration）對結果的影響

ε = 0.1 的探索率對兩種演算法產生**截然不同的影響**：

```
對 Q-Learning：
  探索 = 「噪音」
  ε-greedy 的隨機行動會帶來額外的 -100 懲罰
  但這個懲罰不影響 Q 值更新（仍用 max Q）
  → 若減小 ε，Q-Learning 表現接近 SARSA（因探索減少，路徑風險降低）
  → 若 ε = 0，Q-Learning 直接沿懸崖走，但沒有探索風險

對 SARSA：
  探索 = 「訓練信號」
  ε-greedy 的隨機行動產生的懲罰被納入 Q 值更新
  → 策略自動學會「考慮探索風險」的安全行為
  → 若減小 ε，SARSA 策略也會向懸崖邊移動（因探索風險降低）
  → 若 ε = 0，SARSA 與 Q-Learning 收斂至相同的最優策略
```

---

## 八、理論比較

### 8.1 Off-Policy vs On-Policy 完整對照

| 面向 | Q-Learning (Off-Policy) | SARSA (On-Policy) |
|------|------------------------|-------------------|
| **更新目標** | $\max_{a'} Q(s', a')$ | $Q(s', a')$，a' 由 ε-greedy 選出 |
| **策略一致性** | 行為策略 ≠ 目標策略 | 行為策略 = 目標策略 |
| **Q值收斂至** | 最優策略 $\pi^*$ 的值 | ε-greedy 策略下的最優值 |
| **Cliff Walking 表現** | 懸崖邊路徑（理論最優）| 安全繞行路徑 |
| **訓練穩定性** | 較低（受探索波動影響） | 較高（探索成本納入更新） |
| **適用情境** | 學習後以貪婪策略部署 | 學習過程中需保持安全 |

### 8.2 更新公式對比

```
Q-Learning：
  TD Target = r + γ · max Q(s', *)
              ┗━━━━━━━━ 與實際行動無關 ━━━━━━━━━━━━━━━┛

SARSA：
  TD Target = r + γ · Q(s', a')   ← a' = ε-greedy(s')
              ┗━━━━ 反映實際選擇的行動 ━━━━━━━━━━━━━━━┛
```

---

## 九、結論

### 9.1 比較摘要

| 比較面向 | 結論 |
|---------|------|
| **收斂較快** | ✅ **SARSA**（~206 回合 vs ~500 回合） |
| **訓練較穩定** | ✅ **SARSA**（標準差為 Q-Learning 的約 1/3） |
| **理論最優策略** | ✅ **Q-Learning**（懸崖邊最短路徑） |
| **安全策略** | ✅ **SARSA**（自動學會遠離懸崖） |

### 9.2 選擇建議

```
選擇 Q-Learning 的情境：
  ├─ 學習完成後以純貪婪（ε=0）策略部署
  ├─ 環境中探索的代價可接受（失誤不致命）
  └─ 需要找到理論上的最優策略

選擇 SARSA 的情境：
  ├─ 學習過程中必須保持安全（如機器人控制）
  ├─ 探索失誤代價極高（如自駕車、醫療決策）
  └─ 需要穩定、可預測的訓練過程
```

### 9.3 核心結論

> 在 ε-greedy 探索環境下，**SARSA 因同策略特性而更能感知並規避探索風險**，在 Cliff Walking 此類邊界敏感環境中，展現更快的收斂速度與更高的訓練穩定性。**Q-Learning 雖學到理論最優策略，但訓練期間風險較高**。若探索率 ε→0，兩者將收斂至相同的最優策略。

---

## 十、執行方式

### 環境需求

```bash
pip install numpy matplotlib
```

### 執行程式

```bash
python cliff_walking.py
```

執行約 30～60 秒後，自動生成以下輸出：

| 輸出 | 說明 |
|------|------|
| `learning_curves.png` | 學習曲線（原始 + 平滑，含收斂標記線） |
| `policy_visualization.png` | 策略箭頭圖與最優路徑 |
| `stability_analysis.png` | 滾動標準差 + 跨實驗報酬分布 |
| `summary_dashboard.png` | 深色主題整合分析儀表板 |

並於終端機輸出統計報告。

### 超參數調整

在 `cliff_walking.py` 最底部修改：

```python
EPISODES  = 500    # 訓練回合數（建議 ≥ 500）
ALPHA     = 0.1    # 學習率（0 < α ≤ 1）
GAMMA     = 0.9    # 折扣因子（0 ≤ γ < 1）
EPSILON   = 0.1    # 探索率（建議 0.05 ～ 0.2）
N_RUNS    = 50     # 重複實驗次數（越多越穩定）
```

---

## 參考資料

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
   - Chapter 6：Temporal-Difference Learning
   - Example 6.6：Cliff Walking
2. Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards*. PhD thesis, University of Cambridge.
3. Rummery, G. A., & Niranjan, M. (1994). *On-Line Q-Learning Using Connectionist Systems*. Technical Report CUED/F-INFENG/TR 166.

---

<div align="center">

Made with ❤️ for Reinforcement Learning Homework

</div>
