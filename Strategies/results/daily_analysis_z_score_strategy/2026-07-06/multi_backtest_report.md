# 📊 MANTRA: Layered Z-Score Session Report (2026-07-06)

* **Strategy Architecture:** `PURE MATHEMATICAL MULTI-SLOT GRID WITH ACTIVE REGIME EXITS`
* **Configured Slot Thresholds:** Slot 1 (`1.5`), Slot 2 (`2.0`), Slot 3 (`2.5`), Slot 4 (`3.0`)
* **Regime Control Thresholds:** Max Slope (`±0.08%`) | Max Entry Dwell (`15m`) | Max Trade Holding (`30m`)
* **Operational Windows:** Entries `04:00 - 20:00` | Forced Hard EOD Close `22:00`

### 🔍 Trend vs. Mean-Reversion Regime Indicators
* **Max Ratio 240m Mean Slope (30m Delta):** `0.1106%`
* **Max Z-Score Dwell Time (|Z| >= 2.0):** `27 minutes`

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 31
* **Batch Win Rate:** 54.84%
* **Pure Combination Trade Yield (Rauw Totaal):** 0.2404%
* **Net Portfolio Session Yield (1x Base Portfolio):** 0.0601%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **0.6009%**
* **Average Yield per Executed Slot (1x Base Portfolio):** 0.0019%
* **Average Yield per Executed Slot (10x Leveraged Portfolio):** 0.0194%

### 📜 Session Transaction Ledger (Slot Decomposition)
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | PnL US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL GOLD | PnL Trade Combination | Cash PnL (1x) | Cash PnL (10x Leverage) | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 05:24 | 05:54 | `SHORT` | 7492.50 | 7486.70 | 0.0774% | `LONG` | 4163.21 | 4162.61 | -0.0144% | **0.0315%** | 0.0079% | **0.0787%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 06:20 | 06:50 | `SHORT` | 7495.30 | 7498.80 | -0.0467% | `LONG` | 4162.91 | 4160.32 | -0.0622% | **-0.0545%** | -0.0136% | **-0.1361%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 06:51 | 07:21 | `SHORT` | 7497.50 | 7498.60 | -0.0147% | `LONG` | 4159.98 | 4150.86 | -0.2192% | **-0.1170%** | -0.0292% | **-0.2924%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 07:05 | 07:21 | `SHORT` | 7498.70 | 7498.60 | 0.0013% | `LONG` | 4154.23 | 4150.86 | -0.0811% | **-0.0399%** | -0.0100% | **-0.0997%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 07:22 | 07:31 | `SHORT` | 7498.00 | 7495.10 | 0.0387% | `LONG` | 4151.22 | 4152.30 | 0.0260% | **0.0323%** | 0.0081% | **0.0809%** | `REGIME_SHIFT_SLOPE_EXIT` |
| **Slot 1** | 08:56 | 09:26 | `SHORT` | 7501.00 | 7503.70 | -0.0360% | `LONG` | 4150.12 | 4151.73 | 0.0388% | **0.0014%** | 0.0003% | **0.0035%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 10:43 | 11:13 | `SHORT` | 7509.50 | 7507.40 | 0.0280% | `LONG` | 4150.44 | 4149.63 | -0.0195% | **0.0042%** | 0.0011% | **0.0106%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 10:44 | 11:13 | `SHORT` | 7509.40 | 7507.40 | 0.0266% | `LONG` | 4149.31 | 4149.63 | 0.0077% | **0.0172%** | 0.0043% | **0.0429%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 10:47 | 11:13 | `SHORT` | 7507.60 | 7507.40 | 0.0027% | `LONG` | 4147.68 | 4149.63 | 0.0470% | **0.0248%** | 0.0062% | **0.0621%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 11:17 | 11:47 | `SHORT` | 7508.60 | 7507.60 | 0.0133% | `LONG` | 4147.50 | 4140.55 | -0.1676% | **-0.0771%** | -0.0193% | **-0.1928%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 11:23 | 11:47 | `SHORT` | 7508.90 | 7507.60 | 0.0173% | `LONG` | 4147.64 | 4140.55 | -0.1709% | **-0.0768%** | -0.0192% | **-0.1920%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 11:26 | 11:47 | `SHORT` | 7509.20 | 7507.60 | 0.0213% | `LONG` | 4144.64 | 4140.55 | -0.0987% | **-0.0387%** | -0.0097% | **-0.0967%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 4** | 11:30 | 11:47 | `SHORT` | 7509.70 | 7507.60 | 0.0280% | `LONG` | 4140.26 | 4140.55 | 0.0070% | **0.0175%** | 0.0044% | **0.0437%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 11:49 | 12:19 | `SHORT` | 7505.10 | 7513.10 | -0.1066% | `LONG` | 4140.50 | 4149.75 | 0.2234% | **0.0584%** | 0.0146% | **0.1460%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 11:52 | 12:19 | `SHORT` | 7510.40 | 7513.10 | -0.0360% | `LONG` | 4142.31 | 4149.75 | 0.1796% | **0.0718%** | 0.0180% | **0.1796%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 14:36 | 15:06 | `SHORT` | 7507.60 | 7505.80 | 0.0240% | `LONG` | 4141.29 | 4146.07 | 0.1154% | **0.0697%** | 0.0174% | **0.1742%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 14:41 | 15:06 | `SHORT` | 7507.20 | 7505.80 | 0.0186% | `LONG` | 4135.07 | 4146.07 | 0.2660% | **0.1423%** | 0.0356% | **0.3558%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 14:41 | 15:06 | `SHORT` | 7507.20 | 7505.80 | 0.0186% | `LONG` | 4135.07 | 4146.07 | 0.2660% | **0.1423%** | 0.0356% | **0.3558%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 15:30 | 15:38 | `SHORT` | 7518.90 | 7503.90 | 0.1995% | `LONG` | 4143.55 | 4146.22 | 0.0644% | **0.1320%** | 0.0330% | **0.3299%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 15:51 | 16:21 | `SHORT` | 7520.80 | 7520.70 | 0.0013% | `LONG` | 4145.33 | 4135.57 | -0.2354% | **-0.1171%** | -0.0293% | **-0.2926%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 16:07 | 16:21 | `SHORT` | 7513.70 | 7520.70 | -0.0932% | `LONG` | 4135.33 | 4135.57 | 0.0058% | **-0.0437%** | -0.0109% | **-0.1092%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 16:07 | 16:21 | `SHORT` | 7513.70 | 7520.70 | -0.0932% | `LONG` | 4135.33 | 4135.57 | 0.0058% | **-0.0437%** | -0.0109% | **-0.1092%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 16:22 | 16:52 | `SHORT` | 7521.70 | 7524.10 | -0.0319% | `LONG` | 4137.29 | 4134.40 | -0.0699% | **-0.0509%** | -0.0127% | **-0.1272%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 16:23 | 16:52 | `SHORT` | 7525.00 | 7524.10 | 0.0120% | `LONG` | 4135.64 | 4134.40 | -0.0300% | **-0.0090%** | -0.0023% | **-0.0225%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 16:23 | 16:52 | `SHORT` | 7525.00 | 7524.10 | 0.0120% | `LONG` | 4135.64 | 4134.40 | -0.0300% | **-0.0090%** | -0.0023% | **-0.0225%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 4** | 16:24 | 16:52 | `SHORT` | 7523.70 | 7524.10 | -0.0053% | `LONG` | 4132.89 | 4134.40 | 0.0365% | **0.0156%** | 0.0039% | **0.0390%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 16:53 | 17:23 | `SHORT` | 7527.40 | 7528.40 | -0.0133% | `LONG` | 4135.95 | 4142.70 | 0.1632% | **0.0750%** | 0.0187% | **0.1874%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 18:14 | 18:37 | `SHORT` | 7537.00 | 7533.10 | 0.0517% | `LONG` | 4140.82 | 4148.87 | 0.1944% | **0.1231%** | 0.0308% | **0.3077%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 19:31 | 20:01 | `LONG` | 7533.00 | 7533.20 | 0.0027% | `SHORT` | 4156.85 | 4160.61 | -0.0905% | **-0.0439%** | -0.0110% | **-0.1097%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 19:34 | 20:01 | `LONG` | 7531.00 | 7533.20 | 0.0292% | `SHORT` | 4158.13 | 4160.61 | -0.0596% | **-0.0152%** | -0.0038% | **-0.0380%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 19:42 | 20:01 | `LONG` | 7525.00 | 7533.20 | 0.1090% | `SHORT` | 4157.54 | 4160.61 | -0.0738% | **0.0176%** | 0.0044% | **0.0439%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
