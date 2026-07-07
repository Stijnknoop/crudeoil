# 📊 MANTRA: Layered Z-Score Session Report (2026-07-06)

* **Strategy Architecture:** `PURE MATHEMATICAL MULTI-SLOT GRID WITH ACTIVE REGIME EXITS`
* **Configured Slot Thresholds:** Slot 1 (`1.5`), Slot 2 (`2.0`), Slot 3 (`2.5`), Slot 4 (`3.0`)
* **Regime Control Thresholds:** Max Slope (`±0.05%`) | Max Entry Dwell (`15m`) | Max Trade Holding (`30m`)
* **Operational Windows:** Entries `04:00 - 20:00` | Forced Hard EOD Close `22:00`

### 🔍 Trend vs. Mean-Reversion Regime Indicators
* **Max Ratio 240m Mean Slope (30m Delta):** `0.1106%`
* **Max Z-Score Dwell Time (|Z| >= 2.0):** `27 minutes`

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 25
* **Batch Win Rate:** 52.00%
* **Pure Combination Trade Yield (Rauw Totaal):** 0.0410%
* **Net Portfolio Session Yield (1x Base Portfolio):** 0.0103%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **0.1026%**
* **Average Yield per Executed Slot (1x Base Portfolio):** 0.0004%
* **Average Yield per Executed Slot (10x Leveraged Portfolio):** 0.0041%

### 📜 Session Transaction Ledger (Slot Decomposition)
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | PnL US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL GOLD | PnL Trade Combination | Cash PnL (1x) | Cash PnL (10x Leverage) | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 05:24 | 05:45 | `SHORT` | 7492.50 | 7490.80 | 0.0227% | `LONG` | 4163.21 | 4164.01 | 0.0192% | **0.0210%** | 0.0052% | **0.0524%** | `REGIME_SHIFT_SLOPE_EXIT` |
| **Slot 1** | 06:20 | 06:50 | `SHORT` | 7495.30 | 7498.80 | -0.0467% | `LONG` | 4162.91 | 4160.32 | -0.0622% | **-0.0545%** | -0.0136% | **-0.1361%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 06:51 | 07:06 | `SHORT` | 7497.50 | 7500.30 | -0.0373% | `LONG` | 4159.98 | 4153.87 | -0.1469% | **-0.0921%** | -0.0230% | **-0.2303%** | `REGIME_SHIFT_SLOPE_EXIT` |
| **Slot 2** | 07:05 | 07:06 | `SHORT` | 7498.70 | 7500.30 | -0.0213% | `LONG` | 4154.23 | 4153.87 | -0.0087% | **-0.0150%** | -0.0038% | **-0.0375%** | `REGIME_SHIFT_SLOPE_EXIT` |
| **Slot 1** | 10:43 | 11:13 | `SHORT` | 7509.50 | 7507.40 | 0.0280% | `LONG` | 4150.44 | 4149.63 | -0.0195% | **0.0042%** | 0.0011% | **0.0106%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 10:44 | 11:13 | `SHORT` | 7509.40 | 7507.40 | 0.0266% | `LONG` | 4149.31 | 4149.63 | 0.0077% | **0.0172%** | 0.0043% | **0.0429%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 10:47 | 11:13 | `SHORT` | 7507.60 | 7507.40 | 0.0027% | `LONG` | 4147.68 | 4149.63 | 0.0470% | **0.0248%** | 0.0062% | **0.0621%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 11:17 | 11:44 | `SHORT` | 7508.60 | 7508.00 | 0.0080% | `LONG` | 4147.50 | 4139.34 | -0.1967% | **-0.0944%** | -0.0236% | **-0.2359%** | `REGIME_SHIFT_SLOPE_EXIT` |
| **Slot 2** | 11:23 | 11:44 | `SHORT` | 7508.90 | 7508.00 | 0.0120% | `LONG` | 4147.64 | 4139.34 | -0.2001% | **-0.0941%** | -0.0235% | **-0.2352%** | `REGIME_SHIFT_SLOPE_EXIT` |
| **Slot 3** | 11:26 | 11:44 | `SHORT` | 7509.20 | 7508.00 | 0.0160% | `LONG` | 4144.64 | 4139.34 | -0.1279% | **-0.0559%** | -0.0140% | **-0.1399%** | `REGIME_SHIFT_SLOPE_EXIT` |
| **Slot 4** | 11:30 | 11:44 | `SHORT` | 7509.70 | 7508.00 | 0.0226% | `LONG` | 4140.26 | 4139.34 | -0.0222% | **0.0002%** | 0.0001% | **0.0005%** | `REGIME_SHIFT_SLOPE_EXIT` |
| **Slot 1** | 14:36 | 15:06 | `SHORT` | 7507.60 | 7505.80 | 0.0240% | `LONG` | 4141.29 | 4146.07 | 0.1154% | **0.0697%** | 0.0174% | **0.1742%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 14:41 | 15:06 | `SHORT` | 7507.20 | 7505.80 | 0.0186% | `LONG` | 4135.07 | 4146.07 | 0.2660% | **0.1423%** | 0.0356% | **0.3558%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 14:41 | 15:06 | `SHORT` | 7507.20 | 7505.80 | 0.0186% | `LONG` | 4135.07 | 4146.07 | 0.2660% | **0.1423%** | 0.0356% | **0.3558%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 15:30 | 15:38 | `SHORT` | 7518.90 | 7503.90 | 0.1995% | `LONG` | 4143.55 | 4146.22 | 0.0644% | **0.1320%** | 0.0330% | **0.3299%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 15:51 | 16:21 | `SHORT` | 7520.80 | 7520.70 | 0.0013% | `LONG` | 4145.33 | 4135.57 | -0.2354% | **-0.1171%** | -0.0293% | **-0.2926%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 16:07 | 16:21 | `SHORT` | 7513.70 | 7520.70 | -0.0932% | `LONG` | 4135.33 | 4135.57 | 0.0058% | **-0.0437%** | -0.0109% | **-0.1092%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 16:07 | 16:21 | `SHORT` | 7513.70 | 7520.70 | -0.0932% | `LONG` | 4135.33 | 4135.57 | 0.0058% | **-0.0437%** | -0.0109% | **-0.1092%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 16:22 | 16:35 | `SHORT` | 7521.70 | 7525.90 | -0.0558% | `LONG` | 4137.29 | 4139.37 | 0.0503% | **-0.0028%** | -0.0007% | **-0.0070%** | `REGIME_SHIFT_SLOPE_EXIT` |
| **Slot 2** | 16:23 | 16:35 | `SHORT` | 7525.00 | 7525.90 | -0.0120% | `LONG` | 4135.64 | 4139.37 | 0.0902% | **0.0391%** | 0.0098% | **0.0978%** | `REGIME_SHIFT_SLOPE_EXIT` |
| **Slot 3** | 16:23 | 16:35 | `SHORT` | 7525.00 | 7525.90 | -0.0120% | `LONG` | 4135.64 | 4139.37 | 0.0902% | **0.0391%** | 0.0098% | **0.0978%** | `REGIME_SHIFT_SLOPE_EXIT` |
| **Slot 4** | 16:24 | 16:35 | `SHORT` | 7523.70 | 7525.90 | -0.0292% | `LONG` | 4132.89 | 4139.37 | 0.1568% | **0.0638%** | 0.0159% | **0.1594%** | `REGIME_SHIFT_SLOPE_EXIT` |
| **Slot 1** | 19:31 | 20:01 | `LONG` | 7533.00 | 7533.20 | 0.0027% | `SHORT` | 4156.85 | 4160.61 | -0.0905% | **-0.0439%** | -0.0110% | **-0.1097%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 19:34 | 20:01 | `LONG` | 7531.00 | 7533.20 | 0.0292% | `SHORT` | 4158.13 | 4160.61 | -0.0596% | **-0.0152%** | -0.0038% | **-0.0380%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 19:42 | 20:01 | `LONG` | 7525.00 | 7533.20 | 0.1090% | `SHORT` | 4157.54 | 4160.61 | -0.0738% | **0.0176%** | 0.0044% | **0.0439%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
