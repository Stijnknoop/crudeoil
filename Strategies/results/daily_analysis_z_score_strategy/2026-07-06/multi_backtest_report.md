# 📊 MANTRA: Layered Z-Score Session Report (2026-07-06)

* **Strategy Architecture:** `PURE MATHEMATICAL MULTI-SLOT GRID WITH ACTIVE REGIME EXITS`
* **Configured Slot Thresholds:** Slot 1 (`1.5`), Slot 2 (`2.0`), Slot 3 (`2.5`), Slot 4 (`3.0`)
* **Regime Control Thresholds:** Max Slope (`±0.08%`) | Max Entry Dwell (`15m`) | Max Trade Holding (`30m`)
* **Operational Windows:** Entries `04:00 - 20:00` | Forced Hard EOD Close `22:00`

### 🔍 Trend vs. Mean-Reversion Regime Indicators
* **Max Ratio 240m Mean Slope (30m Delta):** `0.1106%`
* **Max Z-Score Dwell Time (|Z| >= 2.0):** `27 minutes`

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 8
* **Batch Win Rate:** 87.50%
* **Pure Combination Trade Yield (Rauw Totaal):** 0.2154%
* **Net Portfolio Session Yield (1x Base Portfolio):** 0.0539%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **0.5385%**
* **Average Yield per Executed Slot (1x Base Portfolio):** 0.0067%
* **Average Yield per Executed Slot (10x Leveraged Portfolio):** 0.0673%

### 📜 Session Transaction Ledger (Slot Decomposition)
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | PnL US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL GOLD | PnL Trade Combination | Cash PnL (1x) | Cash PnL (10x Leverage) | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 07:05 | 07:31 | `SHORT` | 7498.70 | 7495.10 | 0.0480% | `LONG` | 4154.23 | 4152.30 | -0.0465% | **0.0008%** | 0.0002% | **0.0019%** | `REGIME_SHIFT_SLOPE_EXIT` |
| **Slot 1** | 11:30 | 12:00 | `SHORT` | 7509.70 | 7513.50 | -0.0506% | `LONG` | 4140.26 | 4144.20 | 0.0952% | **0.0223%** | 0.0056% | **0.0557%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 11:31 | 12:00 | `SHORT` | 7507.70 | 7513.50 | -0.0773% | `LONG` | 4138.66 | 4144.20 | 0.1339% | **0.0283%** | 0.0071% | **0.0708%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 11:31 | 12:00 | `SHORT` | 7507.70 | 7513.50 | -0.0773% | `LONG` | 4138.66 | 4144.20 | 0.1339% | **0.0283%** | 0.0071% | **0.0708%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 16:08 | 16:38 | `SHORT` | 7513.90 | 7523.70 | -0.1304% | `LONG` | 4135.13 | 4138.34 | 0.0776% | **-0.0264%** | -0.0066% | **-0.0660%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 16:18 | 16:38 | `SHORT` | 7519.20 | 7523.70 | -0.0598% | `LONG` | 4135.18 | 4138.34 | 0.0764% | **0.0083%** | 0.0021% | **0.0207%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 16:24 | 16:38 | `SHORT` | 7523.70 | 7523.70 | 0.0000% | `LONG` | 4132.89 | 4138.34 | 0.1319% | **0.0659%** | 0.0165% | **0.1648%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 16:43 | 17:13 | `SHORT` | 7522.80 | 7526.40 | -0.0479% | `LONG` | 4130.39 | 4139.63 | 0.2237% | **0.0879%** | 0.0220% | **0.2198%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
