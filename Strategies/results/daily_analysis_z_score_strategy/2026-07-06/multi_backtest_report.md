# 📊 MANTRA: Layered Z-Score Session Report (2026-07-06)

* **Strategy Architecture:** `PURE MATHEMATICAL MULTI-SLOT GRID WITH ACTIVE REGIME EXITS`
* **Configured Slot Thresholds:** Slot 1 (`1.5`), Slot 2 (`2.0`), Slot 3 (`2.5`), Slot 4 (`3.0`)
* **Regime Control Thresholds:** Max Slope (`±0.08%`) | Max Entry Dwell (`30m`) | Max Trade Holding (`60m`)
* **Operational Windows:** Entries `04:00 - 20:00` | Forced Hard EOD Close `22:00`

### 🔍 Trend vs. Mean-Reversion Regime Indicators
* **Max Ratio 240m Mean Slope (30m Delta):** `0.1106%`
* **Max Z-Score Dwell Time (|Z| >= 2.0):** `27 minutes`

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 24
* **Batch Win Rate:** 62.50%
* **Pure Combination Trade Yield (Rauw Totaal):** 0.5210%
* **Net Portfolio Session Yield (1x Base Portfolio):** 0.1303%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **1.3026%**
* **Average Yield per Executed Slot (1x Base Portfolio):** 0.0054%
* **Average Yield per Executed Slot (10x Leveraged Portfolio):** 0.0543%

### 📜 Session Transaction Ledger (Slot Decomposition)
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | PnL US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL GOLD | PnL Trade Combination | Cash PnL (1x) | Cash PnL (10x Leverage) | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 05:24 | 06:24 | `SHORT` | 7492.50 | 7496.30 | -0.0507% | `LONG` | 4163.21 | 4163.20 | -0.0002% | **-0.0255%** | -0.0064% | **-0.0637%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 06:25 | 07:25 | `SHORT` | 7495.30 | 7499.10 | -0.0507% | `LONG` | 4161.72 | 4153.17 | -0.2054% | **-0.1281%** | -0.0320% | **-0.3202%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 07:05 | 07:25 | `SHORT` | 7498.70 | 7499.10 | -0.0053% | `LONG` | 4154.23 | 4153.17 | -0.0255% | **-0.0154%** | -0.0039% | **-0.0386%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 07:26 | 07:31 | `SHORT` | 7498.70 | 7495.10 | 0.0480% | `LONG` | 4153.30 | 4152.30 | -0.0241% | **0.0120%** | 0.0030% | **0.0299%** | `REGIME_SHIFT_SLOPE_EXIT` |
| **Slot 1** | 08:56 | 09:41 | `SHORT` | 7501.00 | 7507.30 | -0.0840% | `LONG` | 4150.12 | 4161.51 | 0.2744% | **0.0952%** | 0.0238% | **0.2381%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 10:43 | 11:43 | `SHORT` | 7509.50 | 7509.30 | 0.0027% | `LONG` | 4150.44 | 4141.32 | -0.2197% | **-0.1085%** | -0.0271% | **-0.2713%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 10:44 | 11:43 | `SHORT` | 7509.40 | 7509.30 | 0.0013% | `LONG` | 4149.31 | 4141.32 | -0.1926% | **-0.0956%** | -0.0239% | **-0.2390%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 10:47 | 11:43 | `SHORT` | 7507.60 | 7509.30 | -0.0226% | `LONG` | 4147.68 | 4141.32 | -0.1533% | **-0.0880%** | -0.0220% | **-0.2200%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 4** | 11:30 | 11:43 | `SHORT` | 7509.70 | 7509.30 | 0.0053% | `LONG` | 4140.26 | 4141.32 | 0.0256% | **0.0155%** | 0.0039% | **0.0387%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 11:44 | 12:34 | `SHORT` | 7507.40 | 7512.80 | -0.0719% | `LONG` | 4139.64 | 4154.11 | 0.3495% | **0.1388%** | 0.0347% | **0.3470%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 2** | 11:45 | 12:34 | `SHORT` | 7507.20 | 7512.80 | -0.0746% | `LONG` | 4140.34 | 4154.11 | 0.3326% | **0.1290%** | 0.0322% | **0.3225%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 14:36 | 15:07 | `SHORT` | 7507.60 | 7506.60 | 0.0133% | `LONG` | 4141.29 | 4148.15 | 0.1656% | **0.0895%** | 0.0224% | **0.2237%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 2** | 14:41 | 15:07 | `SHORT` | 7507.20 | 7506.60 | 0.0080% | `LONG` | 4135.07 | 4148.15 | 0.3163% | **0.1622%** | 0.0405% | **0.4054%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 3** | 14:41 | 15:07 | `SHORT` | 7507.20 | 7506.60 | 0.0080% | `LONG` | 4135.07 | 4148.15 | 0.3163% | **0.1622%** | 0.0405% | **0.4054%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 15:30 | 15:38 | `SHORT` | 7518.90 | 7503.90 | 0.1995% | `LONG` | 4143.55 | 4146.22 | 0.0644% | **0.1320%** | 0.0330% | **0.3299%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 15:51 | 16:51 | `SHORT` | 7520.80 | 7525.00 | -0.0558% | `LONG` | 4145.33 | 4135.75 | -0.2311% | **-0.1435%** | -0.0359% | **-0.3587%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 16:07 | 16:51 | `SHORT` | 7513.70 | 7525.00 | -0.1504% | `LONG` | 4135.33 | 4135.75 | 0.0102% | **-0.0701%** | -0.0175% | **-0.1753%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 16:07 | 16:51 | `SHORT` | 7513.70 | 7525.00 | -0.1504% | `LONG` | 4135.33 | 4135.75 | 0.0102% | **-0.0701%** | -0.0175% | **-0.1753%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 4** | 16:24 | 16:51 | `SHORT` | 7523.70 | 7525.00 | -0.0173% | `LONG` | 4132.89 | 4135.75 | 0.0692% | **0.0260%** | 0.0065% | **0.0649%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 16:52 | 17:52 | `SHORT` | 7523.70 | 7534.40 | -0.1422% | `LONG` | 4135.66 | 4146.15 | 0.2536% | **0.0557%** | 0.0139% | **0.1393%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 18:14 | 18:37 | `SHORT` | 7537.00 | 7533.10 | 0.0517% | `LONG` | 4140.82 | 4148.87 | 0.1944% | **0.1231%** | 0.0308% | **0.3077%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 19:31 | 20:31 | `LONG` | 7533.00 | 7541.70 | 0.1155% | `SHORT` | 4156.85 | 4160.69 | -0.0924% | **0.0116%** | 0.0029% | **0.0289%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 19:34 | 20:31 | `LONG` | 7531.00 | 7541.70 | 0.1421% | `SHORT` | 4158.13 | 4160.69 | -0.0616% | **0.0403%** | 0.0101% | **0.1006%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 19:42 | 20:31 | `LONG` | 7525.00 | 7541.70 | 0.2219% | `SHORT` | 4157.54 | 4160.69 | -0.0758% | **0.0731%** | 0.0183% | **0.1827%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
