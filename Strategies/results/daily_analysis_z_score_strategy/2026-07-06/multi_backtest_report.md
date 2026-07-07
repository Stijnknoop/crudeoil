# 📊 MANTRA: Layered Z-Score Session Report (2026-07-06)

* **Strategy Architecture:** `PURE MATHEMATICAL MULTI-SLOT GRID WITH 15M Z-SMA FILTER`
* **Configured Slot Thresholds:** Slot 1 (`1.5`), Slot 2 (`2.0`), Slot 3 (`2.5`), Slot 4 (`3.0`)
* **Filters:** Z-Score Smoothing (`15m SMA`) | Max Z-Stop (`3.5`) | Freeze (`120m`)
* **Operational Windows:** Entries `04:00 - 20:00` | Forced Hard EOD Close `22:00`

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 6
* **Batch Win Rate:** 66.67%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **0.7762%**

### 📜 Session Transaction Ledger (Slot Decomposition)
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL Trade Combination | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 06:36 | 09:36 | `SHORT` | 7495.40 | 7505.80 | `LONG` | 4160.37 | 4157.04 | **-0.1094%** | `MAX_HOLDING_TIME_EXCEEDED` |
| **Slot 1** | 11:37 | 12:34 | `SHORT` | 7509.80 | 7512.80 | `LONG` | 4142.33 | 4154.11 | **0.1222%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 2** | 11:52 | 12:34 | `SHORT` | 7510.40 | 7512.80 | `LONG` | 4142.31 | 4154.11 | **0.1265%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 14:46 | 15:07 | `SHORT` | 7507.80 | 7506.60 | `LONG` | 4139.14 | 4148.15 | **0.1168%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 16:11 | 18:37 | `SHORT` | 7515.90 | 7533.10 | `LONG` | 4142.05 | 4148.87 | **-0.0321%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 19:47 | 21:23 | `LONG` | 7530.70 | 7550.30 | `SHORT` | 4158.49 | 4162.12 | **0.0865%** | `MEAN_REVERSION_CONVERGENCE` |
