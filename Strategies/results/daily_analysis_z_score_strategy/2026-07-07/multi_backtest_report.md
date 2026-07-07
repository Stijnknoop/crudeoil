# 📊 MANTRA: Layered Z-Score Session Report (2026-07-07)

* **Strategy Architecture:** `PURE MATHEMATICAL MULTI-SLOT GRID WITH ACTIVE REGIME EXITS`
* **Configured Slot Thresholds:** Slot 1 (`1.5`), Slot 2 (`2.0`), Slot 3 (`2.5`), Slot 4 (`3.0`)
* **Regime Control Thresholds:** Max Slope (`±0.05%`) | Max Entry Dwell (`15m`) | Max Trade Holding (`30m`)
* **Operational Windows:** Entries `04:00 - 20:00` | Forced Hard EOD Close `22:00`

### 🔍 Trend vs. Mean-Reversion Regime Indicators
* **Max Ratio 240m Mean Slope (30m Delta):** `0.1377%`
* **Max Z-Score Dwell Time (|Z| >= 2.0):** `68 minutes`

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 17
* **Batch Win Rate:** 47.06%
* **Pure Combination Trade Yield (Rauw Totaal):** 0.3114%
* **Net Portfolio Session Yield (1x Base Portfolio):** 0.0778%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **0.7784%**
* **Average Yield per Executed Slot (1x Base Portfolio):** 0.0046%
* **Average Yield per Executed Slot (10x Leveraged Portfolio):** 0.0458%

### 📜 Session Transaction Ledger (Slot Decomposition)
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | PnL US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL GOLD | PnL Trade Combination | Cash PnL (1x) | Cash PnL (10x Leverage) | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 07:33 | 08:03 | `SHORT` | 7518.80 | 7523.00 | -0.0559% | `LONG` | 4121.50 | 4126.59 | 0.1235% | **0.0338%** | 0.0085% | **0.0845%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 07:38 | 08:03 | `SHORT` | 7519.70 | 7523.00 | -0.0439% | `LONG` | 4122.27 | 4126.59 | 0.1048% | **0.0305%** | 0.0076% | **0.0761%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 07:38 | 08:03 | `SHORT` | 7519.70 | 7523.00 | -0.0439% | `LONG` | 4122.27 | 4126.59 | 0.1048% | **0.0305%** | 0.0076% | **0.0761%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 4** | 07:39 | 08:03 | `SHORT` | 7520.60 | 7523.00 | -0.0319% | `LONG` | 4120.30 | 4126.59 | 0.1527% | **0.0604%** | 0.0151% | **0.1509%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 10:53 | 11:23 | `SHORT` | 7528.70 | 7531.00 | -0.0305% | `LONG` | 4122.39 | 4126.59 | 0.1019% | **0.0357%** | 0.0089% | **0.0892%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 11:28 | 11:35 | `SHORT` | 7530.50 | 7529.00 | 0.0199% | `LONG` | 4122.29 | 4131.19 | 0.2159% | **0.1179%** | 0.0295% | **0.2948%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 2** | 11:29 | 11:35 | `SHORT` | 7530.10 | 7529.00 | 0.0146% | `LONG` | 4121.85 | 4131.19 | 0.2266% | **0.1206%** | 0.0302% | **0.3015%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 3** | 11:29 | 11:35 | `SHORT` | 7530.10 | 7529.00 | 0.0146% | `LONG` | 4121.85 | 4131.19 | 0.2266% | **0.1206%** | 0.0302% | **0.3015%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 13:02 | 13:32 | `LONG` | 7522.60 | 7522.70 | 0.0013% | `SHORT` | 4135.52 | 4140.50 | -0.1204% | **-0.0595%** | -0.0149% | **-0.1489%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 13:18 | 13:32 | `LONG` | 7527.40 | 7522.70 | -0.0624% | `SHORT` | 4141.93 | 4140.50 | 0.0345% | **-0.0140%** | -0.0035% | **-0.0349%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 13:18 | 13:32 | `LONG` | 7527.40 | 7522.70 | -0.0624% | `SHORT` | 4141.93 | 4140.50 | 0.0345% | **-0.0140%** | -0.0035% | **-0.0349%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 17:56 | 18:26 | `SHORT` | 7495.80 | 7515.60 | -0.2641% | `LONG` | 4136.85 | 4145.36 | 0.2057% | **-0.0292%** | -0.0073% | **-0.0730%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 17:57 | 18:26 | `SHORT` | 7498.40 | 7515.60 | -0.2294% | `LONG` | 4138.39 | 4145.36 | 0.1684% | **-0.0305%** | -0.0076% | **-0.0762%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 17:58 | 18:26 | `SHORT` | 7499.20 | 7515.60 | -0.2187% | `LONG` | 4137.78 | 4145.36 | 0.1832% | **-0.0177%** | -0.0044% | **-0.0444%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 18:27 | 18:57 | `SHORT` | 7515.70 | 7514.40 | 0.0173% | `LONG` | 4146.76 | 4143.84 | -0.0704% | **-0.0266%** | -0.0066% | **-0.0664%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 18:31 | 18:57 | `SHORT` | 7515.60 | 7514.40 | 0.0160% | `LONG` | 4145.13 | 4143.84 | -0.0311% | **-0.0076%** | -0.0019% | **-0.0189%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 18:58 | 19:23 | `SHORT` | 7513.60 | 7517.70 | -0.0546% | `LONG` | 4144.02 | 4143.01 | -0.0244% | **-0.0395%** | -0.0099% | **-0.0987%** | `REGIME_SHIFT_SLOPE_EXIT` |
