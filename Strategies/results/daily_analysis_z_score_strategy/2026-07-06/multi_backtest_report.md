# 📊 MANTRA: Layered Z-Score Session Report (2026-07-06)

* **Strategy Architecture:** `PURE MATHEMATICAL MULTI-SLOT GRID WITH REGIME FILTERS`
* **Configured Slot Thresholds:** Slot 1 (`1.5`), Slot 2 (`2.0`), Slot 3 (`2.5`), Slot 4 (`3.0`)
* **Regime Control Thresholds:** Max Slope (`±0.08%`) | Max Entry Dwell (`30m`) | Exit Dwell (`60m`)
* **Operational Windows:** Entries `04:00 - 20:00` | Forced Hard EOD Close `22:00`

### 🔍 Trend vs. Mean-Reversion Regime Indicators
* **Max Ratio 240m Mean Slope (30m Delta):** `0.1106%`
* **Max Z-Score Dwell Time (|Z| >= 2.0):** `27 minutes`

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 17
* **Batch Win Rate:** 88.24%
* **Pure Combination Trade Yield (Rauw Totaal):** 1.1094%
* **Net Portfolio Session Yield (1x Base Portfolio):** 0.2774%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **2.7736%**
* **Average Yield per Executed Slot (1x Base Portfolio):** 0.0163%
* **Average Yield per Executed Slot (10x Leveraged Portfolio):** 0.1632%

### 📜 Session Transaction Ledger (Slot Decomposition)
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | PnL US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL GOLD | PnL Trade Combination | Cash PnL (1x) | Cash PnL (10x Leverage) | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 05:24 | 09:41 | `SHORT` | 7492.50 | 7507.30 | -0.1975% | `LONG` | 4163.21 | 4161.51 | -0.0408% | **-0.1192%** | -0.0298% | **-0.2980%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 2** | 07:05 | 09:41 | `SHORT` | 7498.70 | 7507.30 | -0.1147% | `LONG` | 4154.23 | 4161.51 | 0.1752% | **0.0303%** | 0.0076% | **0.0757%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 10:43 | 12:34 | `SHORT` | 7509.50 | 7512.80 | -0.0439% | `LONG` | 4150.44 | 4154.11 | 0.0884% | **0.0222%** | 0.0056% | **0.0556%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 2** | 10:44 | 12:34 | `SHORT` | 7509.40 | 7512.80 | -0.0453% | `LONG` | 4149.31 | 4154.11 | 0.1157% | **0.0352%** | 0.0088% | **0.0880%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 3** | 10:47 | 12:34 | `SHORT` | 7507.60 | 7512.80 | -0.0693% | `LONG` | 4147.68 | 4154.11 | 0.1550% | **0.0429%** | 0.0107% | **0.1072%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 4** | 11:30 | 12:34 | `SHORT` | 7509.70 | 7512.80 | -0.0413% | `LONG` | 4140.26 | 4154.11 | 0.3345% | **0.1466%** | 0.0367% | **0.3666%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 14:36 | 15:07 | `SHORT` | 7507.60 | 7506.60 | 0.0133% | `LONG` | 4141.29 | 4148.15 | 0.1656% | **0.0895%** | 0.0224% | **0.2237%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 2** | 14:41 | 15:07 | `SHORT` | 7507.20 | 7506.60 | 0.0080% | `LONG` | 4135.07 | 4148.15 | 0.3163% | **0.1622%** | 0.0405% | **0.4054%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 3** | 14:41 | 15:07 | `SHORT` | 7507.20 | 7506.60 | 0.0080% | `LONG` | 4135.07 | 4148.15 | 0.3163% | **0.1622%** | 0.0405% | **0.4054%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 15:30 | 15:38 | `SHORT` | 7518.90 | 7503.90 | 0.1995% | `LONG` | 4143.55 | 4146.22 | 0.0644% | **0.1320%** | 0.0330% | **0.3299%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 15:51 | 18:37 | `SHORT` | 7520.80 | 7533.10 | -0.1635% | `LONG` | 4145.33 | 4148.87 | 0.0854% | **-0.0391%** | -0.0098% | **-0.0977%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 2** | 16:07 | 18:37 | `SHORT` | 7513.70 | 7533.10 | -0.2582% | `LONG` | 4135.33 | 4148.87 | 0.3274% | **0.0346%** | 0.0087% | **0.0865%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 3** | 16:07 | 18:37 | `SHORT` | 7513.70 | 7533.10 | -0.2582% | `LONG` | 4135.33 | 4148.87 | 0.3274% | **0.0346%** | 0.0087% | **0.0865%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 4** | 16:24 | 18:37 | `SHORT` | 7523.70 | 7533.10 | -0.1249% | `LONG` | 4132.89 | 4148.87 | 0.3867% | **0.1309%** | 0.0327% | **0.3271%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 19:31 | 21:23 | `LONG` | 7533.00 | 7550.30 | 0.2297% | `SHORT` | 4156.85 | 4162.12 | -0.1268% | **0.0514%** | 0.0129% | **0.1286%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 2** | 19:34 | 21:23 | `LONG` | 7531.00 | 7550.30 | 0.2563% | `SHORT` | 4158.13 | 4162.12 | -0.0960% | **0.0802%** | 0.0200% | **0.2004%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 3** | 19:42 | 21:23 | `LONG` | 7525.00 | 7550.30 | 0.3362% | `SHORT` | 4157.54 | 4162.12 | -0.1102% | **0.1130%** | 0.0283% | **0.2826%** | `MEAN_REVERSION_CONVERGENCE` |
