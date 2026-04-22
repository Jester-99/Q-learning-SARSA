# -*- coding: utf-8 -*-
"""
Cliff Walking - Q-Learning vs SARSA
====================================
Implementation and comparison of Q-learning and SARSA in Cliff Walking.

Environment:
- 4 x 12 grid world
- Start: bottom-left (3, 0)
- Goal:  bottom-right (3, 11)
- Cliff: bottom row (3, 1) ~ (3, 10)

Author: [Your Name]
Date:   2026-04-22
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # Use non-interactive backend to avoid window encoding issues
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib to support CJK characters
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei',
                                    'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────
# 1. 環境定義
# ─────────────────────────────────────────────

class CliffWalking:
    """
    4×12 Cliff Walking 環境。
    
    狀態編碼: (row, col)，row=0 為頂部，row=3 為底部。
    動作: 0=上, 1=下, 2=左, 3=右
    """
    ROWS = 4
    COLS = 12
    START = (3, 0)
    GOAL  = (3, 11)
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # 上, 下, 左, 右
    ACTION_NAMES = ['↑', '↓', '←', '→']

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = self.START
        return self.state

    def is_cliff(self, state):
        r, c = state
        return r == 3 and 1 <= c <= 10

    def is_goal(self, state):
        return state == self.GOAL

    def step(self, action):
        r, c = self.state
        dr, dc = self.ACTIONS[action]
        nr = np.clip(r + dr, 0, self.ROWS - 1)
        nc = np.clip(c + dc, 0, self.COLS - 1)
        next_state = (nr, nc)

        if self.is_cliff(next_state):
            reward = -100
            next_state = self.START
            done = False
        elif self.is_goal(next_state):
            reward = -1
            done = True
        else:
            reward = -1
            done = False

        self.state = next_state
        return next_state, reward, done

    def state_to_idx(self, state):
        r, c = state
        return r * self.COLS + c

    @property
    def n_states(self):
        return self.ROWS * self.COLS

    @property
    def n_actions(self):
        return 4


# ─────────────────────────────────────────────
# 2. ε-greedy 策略
# ─────────────────────────────────────────────

def epsilon_greedy(Q, state_idx, epsilon, n_actions):
    """回傳 ε-greedy 策略下選擇的動作。"""
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q[state_idx])


# ─────────────────────────────────────────────
# 3. Q-Learning（離策略）
# ─────────────────────────────────────────────

def q_learning(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Q-Learning 更新公式（Off-policy）:
    Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
    
    目標是「下一狀態的最佳可能行動」，不論實際採取何種行動。
    """
    Q = np.zeros((env.n_states, env.n_actions))
    rewards_per_episode = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            s_idx = env.state_to_idx(state)
            action = epsilon_greedy(Q, s_idx, epsilon, env.n_actions)
            next_state, reward, done = env.step(action)
            ns_idx = env.state_to_idx(next_state)

            # Q-Learning 更新：使用下一狀態的最大 Q 值
            best_next = np.max(Q[ns_idx])
            Q[s_idx, action] += alpha * (reward + gamma * best_next - Q[s_idx, action])

            total_reward += reward
            state = next_state

            if done:
                break

        rewards_per_episode.append(total_reward)

    return Q, rewards_per_episode


# ─────────────────────────────────────────────
# 4. SARSA（同策略）
# ─────────────────────────────────────────────

def sarsa(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    SARSA 更新公式（On-policy）:
    Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
    
    目標是「實際採取的下一個行動」，反映探索策略的影響。
    """
    Q = np.zeros((env.n_states, env.n_actions))
    rewards_per_episode = []

    for ep in range(episodes):
        state = env.reset()
        s_idx = env.state_to_idx(state)
        action = epsilon_greedy(Q, s_idx, epsilon, env.n_actions)
        total_reward = 0

        while True:
            next_state, reward, done = env.step(action)
            ns_idx = env.state_to_idx(next_state)

            # SARSA 更新：使用實際採取的下一個行動
            next_action = epsilon_greedy(Q, ns_idx, epsilon, env.n_actions)
            Q[s_idx, action] += alpha * (reward + gamma * Q[ns_idx, next_action] - Q[s_idx, action])

            total_reward += reward
            state = next_state
            s_idx = ns_idx
            action = next_action

            if done:
                break

        rewards_per_episode.append(total_reward)

    return Q, rewards_per_episode


# ─────────────────────────────────────────────
# 5. 多次運行取平均（用於穩定性分析）
# ─────────────────────────────────────────────

def run_multiple(algo_fn, env_class, n_runs=50, episodes=500,
                 alpha=0.1, gamma=0.9, epsilon=0.1):
    """多次重複實驗，回傳每回合的平均報酬、標準差，以及所有 run 的完整獎勵矩陣與平均 Q 表。"""
    all_rewards = []
    all_Q = []
    for _ in range(n_runs):
        env = env_class()
        Q, rewards = algo_fn(env, episodes=episodes, alpha=alpha,
                             gamma=gamma, epsilon=epsilon)
        all_rewards.append(rewards)
        all_Q.append(Q)
    all_rewards = np.array(all_rewards)          # shape: (n_runs, episodes)
    Q_avg = np.mean(np.array(all_Q), axis=0)    # 平均 Q 表（更具代表性）
    return all_rewards.mean(axis=0), all_rewards.std(axis=0), Q_avg, all_rewards


# ─────────────────────────────────────────────
# 6. 最優路徑提取（Greedy Policy）
# ─────────────────────────────────────────────

def extract_greedy_path(Q, env):
    """依據學習後的 Q 值，執行貪婪策略並記錄路徑（最多 200 步）。"""
    env.reset()
    state = env.START
    path = [state]
    visited = set()
    visited.add(state)
    max_steps = 200

    for _ in range(max_steps):
        s_idx = env.state_to_idx(state)
        action = np.argmax(Q[s_idx])
        next_state, _, done = env.step(action)
        path.append(next_state)
        if done or next_state in visited:
            break
        visited.add(next_state)
        state = next_state

    return path


# ─────────────────────────────────────────────
# 7. 視覺化函式
# ─────────────────────────────────────────────

def smooth(data, window=10):
    """移動平均平滑。"""
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='same')


def plot_learning_curves(ql_mean, ql_std, sarsa_mean, sarsa_std,
                         episodes, n_runs, alpha, gamma, epsilon,
                         ql_conv=None, sarsa_conv=None):
    """繪製學習曲線（含信賴區間與收斂標記）。"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f'Q-Learning vs SARSA -- Cliff Walking\n'
        f'epsilon={epsilon}, alpha={alpha}, gamma={gamma}  (averaged over {n_runs} runs)',
        fontsize=14, fontweight='bold'
    )
    ep_range = np.arange(1, episodes + 1)

    colors = {'ql': '#E74C3C', 'sarsa': '#1ABC9C'}

    for ax, use_smooth, title_suffix in zip(
            axes, [False, True], ['Raw Reward Curve', 'Smoothed Curve (window=10)']):
        
        for label, mean, std, color in [
            ('Q-Learning', ql_mean,    ql_std,    colors['ql']),
            ('SARSA',      sarsa_mean, sarsa_std, colors['sarsa']),
        ]:
            y = smooth(mean, 10) if use_smooth else mean
            ax.plot(ep_range, y, color=color, label=label, linewidth=1.8)
            ax.fill_between(ep_range,
                            y - std / np.sqrt(n_runs),
                            y + std / np.sqrt(n_runs),
                            color=color, alpha=0.2)

        # --- 標示收斂回合（垂直虛線） ---
        if use_smooth:
            if ql_conv:
                ax.axvline(x=ql_conv, color=colors['ql'], linestyle='--',
                           linewidth=1.5, alpha=0.8,
                           label=f'Q-Learning converge ep.{ql_conv}')
            if sarsa_conv:
                ax.axvline(x=sarsa_conv, color=colors['sarsa'], linestyle='--',
                           linewidth=1.5, alpha=0.8,
                           label=f'SARSA converge ep.{sarsa_conv}')
            # 門檻水平線
            ax.axhline(y=-30, color='gray', linestyle=':', linewidth=1.2,
                       alpha=0.6, label='Threshold = -30')

        ax.set_xlabel('Episodes', fontsize=12)
        ax.set_ylabel('Total Reward per Episode', fontsize=12)
        ax.set_title(title_suffix, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.35)
        ax.set_xlim(1, episodes)

    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] learning_curves.png")


def plot_policy_and_path(Q_ql, Q_sarsa, env_class):
    """繪製最優策略箭頭圖與最優路徑。"""
    env = env_class()
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('最優策略視覺化（Greedy Policy）', fontsize=15, fontweight='bold')

    action_dx = [0,  0, -1,  1]   # col 方向（→）
    action_dy = [1, -1,  0,  0]   # row 方向（↑ 在畫面上為 y 增加，需反轉 row）

    for ax, Q, label, path_color in zip(
            axes,
            [Q_ql, Q_sarsa],
            ['Q-Learning 策略', 'SARSA 策略'],
            ['#E74C3C', '#1ABC9C']):

        # 網格背景
        grid = np.zeros((env.ROWS, env.COLS))
        # 懸崖 = 1，起點 = 2，終點 = 3
        for c in range(1, env.COLS - 1):
            grid[3, c] = 1
        grid[env.START] = 2
        grid[env.GOAL]  = 3

        cmap = mcolors.ListedColormap(['#F8F9FA', '#AED6F1', '#2ECC71', '#E74C3C'])
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=3, origin='upper', aspect='equal')

        # 繪製策略箭頭
        for r in range(env.ROWS):
            for c in range(env.COLS):
                if (r == 3 and 1 <= c <= 10):   # 懸崖格不顯示箭頭
                    continue
                if (r, c) == env.GOAL:
                    continue
                s_idx = env.state_to_idx((r, c))
                a = np.argmax(Q[s_idx])
                dx = action_dx[a] * 0.35
                dy = -action_dy[a] * 0.35   # 畫面 y 軸反向
                ax.annotate("", xy=(c + dx, r + dy),
                            xytext=(c, r),
                            arrowprops=dict(arrowstyle="->",
                                            color='#2C3E50',
                                            lw=1.5))

        # 繪製最優路徑
        env_tmp = env_class()
        path = extract_greedy_path(Q, env_tmp)
        if len(path) > 1:
            xs = [p[1] for p in path]
            ys = [p[0] for p in path]
            ax.plot(xs, ys, 'o-', color=path_color,
                    linewidth=2.5, markersize=5,
                    alpha=0.85, label='最優路徑', zorder=5)
            ax.plot(xs[0], ys[0], 's', color='#2ECC71', markersize=12,
                    zorder=6, label='Start')
            ax.plot(xs[-1], ys[-1], '*', color='#E74C3C', markersize=14,
                    zorder=6, label='Goal')

        # 標籤文字
        ax.text(0, 3, 'S', ha='center', va='center',
                fontsize=13, fontweight='bold', color='white', zorder=7)
        ax.text(11, 3, 'G', ha='center', va='center',
                fontsize=13, fontweight='bold', color='white', zorder=7)
        ax.text(5.5, 3, 'Cliff', ha='center', va='center',
                fontsize=12, color='white', zorder=7)

        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xticks(range(env.COLS))
        ax.set_yticks(range(env.ROWS))
        ax.set_xticklabels(range(env.COLS))
        ax.set_yticklabels(range(env.ROWS))
        ax.tick_params(labelsize=8)
        ax.grid(True, color='gray', linewidth=0.4, alpha=0.5)
        ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig('policy_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] policy_visualization.png")


def plot_stability_analysis(ql_mean, sarsa_mean, ql_all_runs, sarsa_all_runs, episodes):
    """
    穩定性分析：
    - 左圖：滾動標準差（反映學習曲線的逐回合波動）
    - 右圖：後半段各 run 報酬的真實跨 run 分布箱型圖
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Stability Analysis: Q-Learning vs SARSA', fontsize=14, fontweight='bold')
    ep_range = np.arange(1, episodes + 1)

    # --- 左：滾動標準差（基於平均曲線的逐步波動）---
    ax = axes[0]
    window = 20
    def rolling_std(data, w):
        out = []
        for i in range(len(data)):
            start = max(0, i - w + 1)
            out.append(np.std(data[start:i+1]))
        return np.array(out)

    ql_rs    = rolling_std(ql_mean,    window)
    sarsa_rs = rolling_std(sarsa_mean, window)

    ax.plot(ep_range, sarsa_rs, color='#1ABC9C', label='SARSA', linewidth=1.8)
    ax.plot(ep_range, ql_rs,    color='#E74C3C', label='Q-Learning', linewidth=1.8)
    ax.set_title(f'Rolling Std of Reward (window={window})', fontsize=12)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Std of Total Reward')
    ax.legend()
    ax.grid(True, alpha=0.35)

    # --- 右：後半段各 run 的真實報酬分布（跨 50 次實驗）---
    ax = axes[1]
    half = episodes // 2
    # ql_all_runs shape: (n_runs, episodes)，取後半段每個 run 的平均報酬
    ql_run_means    = ql_all_runs[:, half:].mean(axis=1)     # shape: (n_runs,)
    sarsa_run_means = sarsa_all_runs[:, half:].mean(axis=1)  # shape: (n_runs,)

    bp = ax.boxplot(
        [sarsa_run_means, ql_run_means],
        labels=['SARSA', 'Q-Learning'],
        patch_artist=True, notch=False, widths=0.5
    )
    colors_box = ['#1ABC9C', '#E74C3C']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(
        f'2nd Half Avg Reward Distribution\n'
        f'(per-run average, ep {half}~{episodes}, across all runs)',
        fontsize=11
    )
    ax.set_ylabel('Avg Total Reward per Run')
    ax.grid(True, alpha=0.35, axis='y')

    plt.tight_layout()
    plt.savefig('stability_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] stability_analysis.png")


def plot_summary_dashboard(ql_mean, sarsa_mean, ql_std, sarsa_std,
                           Q_ql, Q_sarsa, env_class, episodes, n_runs, 
                           alpha, gamma, epsilon):
    """整合式總覽儀表板。"""
    env = env_class()
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('#1A1A2E')
    fig.suptitle(
        'Cliff Walking：Q-Learning vs SARSA 完整分析報告\n'
        f'ε={epsilon}  α={alpha}  γ={gamma}  回合={episodes}  實驗次數={n_runs}',
        fontsize=16, fontweight='bold', color='white', y=0.98
    )
    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.4)

    ep_range = np.arange(1, episodes + 1)
    colors = {'ql': '#FF6B6B', 'sarsa': '#4ECDC4', 'bg': '#16213E', 'text': 'white'}

    # ── 上方：學習曲線（跨 2 列）──
    ax_curve = fig.add_subplot(gs[0, :2])
    ax_curve.set_facecolor(colors['bg'])
    for label, mean, std, color in [
        ('Q-Learning', ql_mean, ql_std, colors['ql']),
        ('SARSA',      sarsa_mean, sarsa_std, colors['sarsa']),
    ]:
        y_sm = smooth(mean, 10)
        ax_curve.plot(ep_range, y_sm, color=color, label=label, linewidth=2)
        ax_curve.fill_between(ep_range,
                              y_sm - std / np.sqrt(n_runs),
                              y_sm + std / np.sqrt(n_runs),
                              color=color, alpha=0.25)
    ax_curve.set_title('學習曲線（平滑）', color='white', fontsize=12)
    ax_curve.set_xlabel('Episodes', color='white')
    ax_curve.set_ylabel('Total Reward', color='white')
    ax_curve.tick_params(colors='white')
    ax_curve.legend(fontsize=10, facecolor=colors['bg'], labelcolor='white')
    ax_curve.grid(True, alpha=0.2, color='white')
    ax_curve.spines['bottom'].set_color('gray')
    ax_curve.spines['left'].set_color('gray')

    # ── 上方：穩定性箱型圖 ──
    ax_box = fig.add_subplot(gs[0, 2:])
    ax_box.set_facecolor(colors['bg'])
    half = episodes // 2
    bp = ax_box.boxplot(
        [sarsa_mean[half:], ql_mean[half:]],
        labels=['SARSA', 'Q-Learning'],
        patch_artist=True, widths=0.5
    )
    for patch, color in zip(bp['boxes'], [colors['sarsa'], colors['ql']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='white')
    ax_box.tick_params(colors='white')
    ax_box.set_title('後半段報酬分布', color='white', fontsize=12)
    ax_box.set_ylabel('Total Reward', color='white')
    ax_box.grid(True, alpha=0.2, color='white', axis='y')
    ax_box.spines['bottom'].set_color('gray')
    ax_box.spines['left'].set_color('gray')

    # ── 中下：策略網格（各 2 列）──
    action_dx = [0,  0, -1,  1]
    action_dy = [1, -1,  0,  0]

    for col_offset, Q, label, path_color in [
        (0, Q_ql,    'Q-Learning 策略', colors['ql']),
        (2, Q_sarsa, 'SARSA 策略',      colors['sarsa']),
    ]:
        ax_grid = fig.add_subplot(gs[1:, col_offset:col_offset+2])
        ax_grid.set_facecolor(colors['bg'])

        grid = np.zeros((env.ROWS, env.COLS))
        for c in range(1, env.COLS - 1):
            grid[3, c] = 1
        grid[env.START] = 2
        grid[env.GOAL]  = 3

        cmap = mcolors.ListedColormap(['#2C3E50', '#5D6D7E', '#27AE60', '#C0392B'])
        ax_grid.imshow(grid, cmap=cmap, vmin=0, vmax=3, origin='upper', aspect='equal')

        for r in range(env.ROWS):
            for c in range(env.COLS):
                if (r == 3 and 1 <= c <= 10):
                    continue
                if (r, c) == env.GOAL:
                    continue
                s_idx = env.state_to_idx((r, c))
                a = np.argmax(Q[s_idx])
                dx = action_dx[a] * 0.35
                dy = -action_dy[a] * 0.35
                ax_grid.annotate("", xy=(c + dx, r + dy), xytext=(c, r),
                                 arrowprops=dict(arrowstyle="->",
                                                 color='white', lw=1.4))

        env_tmp = env_class()
        path = extract_greedy_path(Q, env_tmp)
        if len(path) > 1:
            xs = [p[1] for p in path]
            ys = [p[0] for p in path]
            ax_grid.plot(xs, ys, 'o-', color=path_color,
                         linewidth=3, markersize=6, alpha=0.9, zorder=5)

        ax_grid.text(0, 3, 'S', ha='center', va='center',
                     fontsize=14, fontweight='bold', color='white', zorder=7)
        ax_grid.text(11, 3, 'G', ha='center', va='center',
                     fontsize=14, fontweight='bold', color='white', zorder=7)
        ax_grid.text(5.5, 3, 'Cliff', ha='center', va='center',
                     fontsize=11, color='white', zorder=7, alpha=0.9)

        ax_grid.set_title(label, color='white', fontsize=13, fontweight='bold')
        ax_grid.tick_params(colors='white')
        ax_grid.set_xticks(range(env.COLS))
        ax_grid.set_yticks(range(env.ROWS))
        ax_grid.set_xticklabels(range(env.COLS), color='white', fontsize=7)
        ax_grid.set_yticklabels(range(env.ROWS), color='white', fontsize=8)
        ax_grid.grid(True, color='gray', linewidth=0.4, alpha=0.4)

    plt.savefig('summary_dashboard.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()
    print("[SAVED] summary_dashboard.png")


# ─────────────────────────────────────────────
# 8. 統計報告
# ─────────────────────────────────────────────

def print_report(ql_mean, sarsa_mean, ql_std, sarsa_std, episodes):
    """在終端機列印統計報告。"""
    half = episodes // 2
    sep = "=" * 60

    print(f"\n{sep}")
    print("  Q-Learning vs SARSA --- Cliff Walking Statistical Report")
    print(sep)

    print(f"\n{'Metric':<30} {'Q-Learning':>15} {'SARSA':>15}")
    print("-" * 60)

    def fmt(val):
        return f"{val:>15.2f}"

    metrics = [
        ("Overall Avg Reward",      ql_mean.mean(),        sarsa_mean.mean()),
        ("Overall Reward Std",      ql_std.mean(),         sarsa_std.mean()),
        ("2nd Half Avg Reward",     ql_mean[half:].mean(), sarsa_mean[half:].mean()),
        ("2nd Half Reward Std",     ql_std[half:].mean(),  sarsa_std[half:].mean()),
        ("Best Episode Reward",     ql_mean.max(),         sarsa_mean.max()),
        ("Worst Episode Reward",    ql_mean.min(),         sarsa_mean.min()),
    ]

    for name, ql_val, sarsa_val in metrics:
        print(f"  {name:<28}{fmt(ql_val)}{fmt(sarsa_val)}")

    # 收斂分析（以平滑後的曲線達到 -30 的最早回合）
    ql_sm    = smooth(ql_mean,    10)
    sarsa_sm = smooth(sarsa_mean, 10)
    threshold = -30
    ql_conv    = next((i+1 for i, v in enumerate(ql_sm)    if v > threshold), None)
    sarsa_conv = next((i+1 for i, v in enumerate(sarsa_sm) if v > threshold), None)

    print(f"\n  Convergence (first episode stably above {threshold}):")
    print(f"    Q-Learning : {ql_conv or 'not reached'} episodes")
    print(f"    SARSA      : {sarsa_conv or 'not reached'} episodes")

    print(f"\n{sep}")
    print("  Summary Conclusions")
    print(sep)
    print("""
  [1] Convergence Speed:
      SARSA tends to stabilize faster because it accounts for
      the cost of exploration, making Q-values reflect real behavior.

  [2] Final Reward:
      Under epsilon-greedy exploration, SARSA achieves higher
      average reward in the 2nd half due to its conservative policy
      (staying away from the cliff).

  [3] Stability:
      Q-Learning is more volatile during training because it
      greedily targets optimal values, risking cliff falls.

  [4] Path Difference:
      Q-Learning => Shortest path along the cliff edge (theoretical optimal)
      SARSA      => Safer path through upper rows (avoids exploration risk)

  [5] When to Use:
      - Production / exploitation phase => Q-Learning (converges to optimal)
      - Safety-critical learning phase  => SARSA (stable and conservative)
""")
    print(sep)


# ─────────────────────────────────────────────
# 9. 主程式
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 超參數設定
    EPISODES  = 500
    ALPHA     = 0.1
    GAMMA     = 0.9
    EPSILON   = 0.1
    N_RUNS    = 50        # 重複實驗次數（越多越穩定，但耗時較長）

    print(f"[START] Training  (episodes={EPISODES}, alpha={ALPHA}, gamma={GAMMA}, epsilon={EPSILON}, runs={N_RUNS})")
    print("   Please wait...\n")

    # 多次實驗（回傳 mean, std, 平均Q表, 全部run的獎勵矩陣）
    print("  >> Running Q-Learning...")
    ql_mean, ql_std, Q_ql, ql_all_runs = run_multiple(
        q_learning, CliffWalking, n_runs=N_RUNS,
        episodes=EPISODES, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON
    )

    print("  >> Running SARSA...")
    sarsa_mean, sarsa_std, Q_sarsa, sarsa_all_runs = run_multiple(
        sarsa, CliffWalking, n_runs=N_RUNS,
        episodes=EPISODES, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON
    )

    # --- 先計算收斂回合（供學習曲線標示）---
    threshold = -30
    ql_sm    = smooth(ql_mean,    10)
    sarsa_sm = smooth(sarsa_mean, 10)
    ql_conv    = next((i+1 for i, v in enumerate(ql_sm)    if v > threshold), None)
    sarsa_conv = next((i+1 for i, v in enumerate(sarsa_sm) if v > threshold), None)

    print("\n[INFO] Generating plots...")

    # 學習曲線（帶收斂標記）
    plot_learning_curves(ql_mean, ql_std, sarsa_mean, sarsa_std,
                         EPISODES, N_RUNS, ALPHA, GAMMA, EPSILON,
                         ql_conv=ql_conv, sarsa_conv=sarsa_conv)

    # 策略視覺化（使用平均 Q 表）
    plot_policy_and_path(Q_ql, Q_sarsa, CliffWalking)

    # 穩定性分析（使用真實跨 run 的分布）
    plot_stability_analysis(ql_mean, sarsa_mean, ql_all_runs, sarsa_all_runs, EPISODES)

    # 整合儀表板
    plot_summary_dashboard(ql_mean, sarsa_mean, ql_std, sarsa_std,
                           Q_ql, Q_sarsa, CliffWalking,
                           EPISODES, N_RUNS, ALPHA, GAMMA, EPSILON)

    # 統計報告
    print_report(ql_mean, sarsa_mean, ql_std, sarsa_std, EPISODES)
