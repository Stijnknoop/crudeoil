# 📊 MANTRA: Layered Z-Score Session Report (2026-07-07)

* **Strategy Architecture:** `PURE MATHEMATICAL MULTI-SLOT GRID WITH ACTIVE REGIME EXITS`
* **Configured Slot Thresholds:** Slot 1 (`1.5`), Slot 2 (`2.0`), Slot 3 (`2.5`), Slot 4 (`3.0`)
* **Regime Control Thresholds:** Max Slope (`±0.08%`) | Max Entry Dwell (`30m`) | Max Trade Holding (`60m`)
* **Operational Windows:** Entries `04:00 - 20:00` | Forced Hard EOD Close `22:00`

### 🔍 Trend vs. Mean-Reversion Regime Indicators
* **Max Ratio 240m Mean Slope (30m Delta):** `0.1377%`
* **Max Z-Score Dwell Time (|Z| >= 2.0):** `68 minutes`

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 17
* **Batch Win Rate:** 52.94%
* **Pure Combination Trade Yield (Rauw Totaal):** -0.0574%
* **Net Portfolio Session Yield (1x Base Portfolio):** -0.0143%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **-0.1435%**
* **Average Yield per Executed Slot (1x Base Portfolio):** -0.0008%
* **Average Yield per Executed Slot (10x Leveraged Portfolio):** -0.0084%

### 📜 Session Transaction Ledger (Slot Decomposition)
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | PnL US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL GOLD | PnL Trade Combination | Cash PnL (1x) | Cash PnL (10x Leverage) | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 04:05 | 05:05 | `SHORT` | 7534.90 | 7532.50 | 0.0319% | `LONG` | 4139.32 | 4138.93 | -0.0094% | **0.0112%** | 0.0028% | **0.0280%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 05:46 | 06:08 | `SHORT` | 7523.40 | 7524.80 | -0.0186% | `LONG` | 4126.76 | 4137.46 | 0.2593% | **0.1203%** | 0.0301% | **0.3008%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 07:33 | 08:33 | `SHORT` | 7518.80 | 7521.00 | -0.0293% | `LONG` | 4121.50 | 4129.17 | 0.1861% | **0.0784%** | 0.0196% | **0.1960%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 2** | 07:38 | 08:33 | `SHORT` | 7519.70 | 7521.00 | -0.0173% | `LONG` | 4122.27 | 4129.17 | 0.1674% | **0.0750%** | 0.0188% | **0.1876%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 3** | 07:38 | 08:33 | `SHORT` | 7519.70 | 7521.00 | -0.0173% | `LONG` | 4122.27 | 4129.17 | 0.1674% | **0.0750%** | 0.0188% | **0.1876%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 4** | 07:39 | 08:33 | `SHORT` | 7520.60 | 7521.00 | -0.0053% | `LONG` | 4120.30 | 4129.17 | 0.2153% | **0.1050%** | 0.0262% | **0.2624%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 10:53 | 11:35 | `SHORT` | 7528.70 | 7529.00 | -0.0040% | `LONG` | 4122.39 | 4131.19 | 0.2135% | **0.1047%** | 0.0262% | **0.2619%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 2** | 11:28 | 11:35 | `SHORT` | 7530.50 | 7529.00 | 0.0199% | `LONG` | 4122.29 | 4131.19 | 0.2159% | **0.1179%** | 0.0295% | **0.2948%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 3** | 11:28 | 11:35 | `SHORT` | 7530.50 | 7529.00 | 0.0199% | `LONG` | 4122.29 | 4131.19 | 0.2159% | **0.1179%** | 0.0295% | **0.2948%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 13:02 | 14:02 | `LONG` | 7522.60 | 7529.10 | 0.0864% | `SHORT` | 4135.52 | 4157.39 | -0.5288% | **-0.2212%** | -0.0553% | **-0.5530%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 13:18 | 14:02 | `LONG` | 7527.40 | 7529.10 | 0.0226% | `SHORT` | 4141.93 | 4157.39 | -0.3733% | **-0.1753%** | -0.0438% | **-0.4383%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 13:18 | 14:02 | `LONG` | 7527.40 | 7529.10 | 0.0226% | `SHORT` | 4141.93 | 4157.39 | -0.3733% | **-0.1753%** | -0.0438% | **-0.4383%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 4** | 13:35 | 14:02 | `LONG` | 7524.50 | 7529.10 | 0.0611% | `SHORT` | 4146.77 | 4157.39 | -0.2561% | **-0.0975%** | -0.0244% | **-0.2437%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 17:51 | 18:51 | `SHORT` | 7496.60 | 7517.00 | -0.2721% | `LONG` | 4141.34 | 4145.46 | 0.0995% | **-0.0863%** | -0.0216% | **-0.2158%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 17:54 | 18:51 | `SHORT` | 7496.30 | 7517.00 | -0.2761% | `LONG` | 4138.18 | 4145.46 | 0.1759% | **-0.0501%** | -0.0125% | **-0.1253%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 17:58 | 18:51 | `SHORT` | 7499.20 | 7517.00 | -0.2374% | `LONG` | 4137.78 | 4145.46 | 0.1856% | **-0.0259%** | -0.0065% | **-0.0647%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 18:52 | 19:52 | `SHORT` | 7516.80 | 7518.30 | -0.0200% | `LONG` | 4145.91 | 4144.14 | -0.0427% | **-0.0313%** | -0.0078% | **-0.0783%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
