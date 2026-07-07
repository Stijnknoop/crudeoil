# 📊 MANTRA: Layered Z-Score Session Report (2026-07-07)

* **Strategy Architecture:** `PURE MATHEMATICAL MULTI-SLOT GRID WITH ACTIVE REGIME EXITS`
* **Configured Slot Thresholds:** Slot 1 (`1.5`), Slot 2 (`2.0`), Slot 3 (`2.5`), Slot 4 (`3.0`)
* **Regime Control Thresholds:** Max Slope (`±0.08%`) | Max Entry Dwell (`15m`) | Max Trade Holding (`30m`)
* **Operational Windows:** Entries `04:00 - 20:00` | Forced Hard EOD Close `22:00`

### 🔍 Trend vs. Mean-Reversion Regime Indicators
* **Max Ratio 240m Mean Slope (30m Delta):** `0.1377%`
* **Max Z-Score Dwell Time (|Z| >= 2.0):** `68 minutes`

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 20
* **Batch Win Rate:** 55.00%
* **Pure Combination Trade Yield (Rauw Totaal):** 0.4428%
* **Net Portfolio Session Yield (1x Base Portfolio):** 0.1107%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **1.1069%**
* **Average Yield per Executed Slot (1x Base Portfolio):** 0.0055%
* **Average Yield per Executed Slot (10x Leveraged Portfolio):** 0.0553%

### 📜 Session Transaction Ledger (Slot Decomposition)
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | PnL US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL GOLD | PnL Trade Combination | Cash PnL (1x) | Cash PnL (10x Leverage) | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 04:05 | 04:35 | `SHORT` | 7534.90 | 7531.70 | 0.0425% | `LONG` | 4139.32 | 4138.53 | -0.0191% | **0.0117%** | 0.0029% | **0.0292%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 04:51 | 05:21 | `SHORT` | 7530.40 | 7529.90 | 0.0066% | `LONG` | 4133.05 | 4137.29 | 0.1026% | **0.0546%** | 0.0137% | **0.1365%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 05:46 | 06:08 | `SHORT` | 7523.40 | 7524.80 | -0.0186% | `LONG` | 4126.76 | 4137.46 | 0.2593% | **0.1203%** | 0.0301% | **0.3008%** | `MEAN_REVERSION_CONVERGENCE` |
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
| **Slot 1** | 17:51 | 18:21 | `SHORT` | 7496.60 | 7511.10 | -0.1934% | `LONG` | 4141.34 | 4143.68 | 0.0565% | **-0.0685%** | -0.0171% | **-0.1711%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 17:54 | 18:21 | `SHORT` | 7496.30 | 7511.10 | -0.1974% | `LONG` | 4138.18 | 4143.68 | 0.1329% | **-0.0323%** | -0.0081% | **-0.0807%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 17:58 | 18:21 | `SHORT` | 7499.20 | 7511.10 | -0.1587% | `LONG` | 4137.78 | 4143.68 | 0.1426% | **-0.0080%** | -0.0020% | **-0.0201%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 18:22 | 18:52 | `SHORT` | 7513.00 | 7517.20 | -0.0559% | `LONG` | 4145.47 | 4145.61 | 0.0034% | **-0.0263%** | -0.0066% | **-0.0657%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 18:23 | 18:52 | `SHORT` | 7513.50 | 7517.20 | -0.0492% | `LONG` | 4145.39 | 4145.61 | 0.0053% | **-0.0220%** | -0.0055% | **-0.0549%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 18:53 | 19:23 | `SHORT` | 7515.60 | 7517.70 | -0.0279% | `LONG` | 4145.94 | 4143.01 | -0.0707% | **-0.0493%** | -0.0123% | **-0.1233%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
