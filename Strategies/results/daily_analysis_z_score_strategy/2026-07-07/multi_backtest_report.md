# 📊 MANTRA: Layered Z-Score Session Report (2026-07-07)

* **Strategy Architecture:** `PURE MATHEMATICAL MULTI-SLOT GRID`
* **Configured Slot Thresholds:** Slot 1 (`1.5`), Slot 2 (`2.0`), Slot 3 (`2.5`), Slot 4 (`3.0`)
* **Operational Windows:** Entries `04:00 - 20:00` | Forced Hard EOD Close `22:00`

### 🔍 Trend vs. Mean-Reversion Regime Indicators
* **Max Ratio 240m Mean Slope (30m Delta):** `0.1377%`
* **Max Z-Score Dwell Time (|Z| >= 2.0):** `68 minutes`

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 15
* **Batch Win Rate:** 53.33%
* **Pure Combination Trade Yield (Rauw Totaal):** -1.4034%
* **Net Portfolio Session Yield (1x Base Portfolio):** -0.3508%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **-3.5084%**
* **Average Yield per Executed Slot (1x Base Portfolio):** -0.0234%
* **Average Yield per Executed Slot (10x Leveraged Portfolio):** -0.2339%

### 📜 Session Transaction Ledger (Slot Decomposition)
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | PnL US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL GOLD | PnL Trade Combination | Cash PnL (1x) | Cash PnL (10x Leverage) | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 04:05 | 06:08 | `SHORT` | 7534.90 | 7524.80 | 0.1340% | `LONG` | 4139.32 | 4137.46 | -0.0449% | **0.0446%** | 0.0111% | **0.1114%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 07:33 | 08:33 | `SHORT` | 7518.80 | 7521.00 | -0.0293% | `LONG` | 4121.50 | 4129.17 | 0.1861% | **0.0784%** | 0.0196% | **0.1960%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 2** | 07:38 | 08:33 | `SHORT` | 7519.70 | 7521.00 | -0.0173% | `LONG` | 4122.27 | 4129.17 | 0.1674% | **0.0750%** | 0.0188% | **0.1876%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 3** | 07:38 | 08:33 | `SHORT` | 7519.70 | 7521.00 | -0.0173% | `LONG` | 4122.27 | 4129.17 | 0.1674% | **0.0750%** | 0.0188% | **0.1876%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 4** | 07:39 | 08:33 | `SHORT` | 7520.60 | 7521.00 | -0.0053% | `LONG` | 4120.30 | 4129.17 | 0.2153% | **0.1050%** | 0.0262% | **0.2624%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 10:53 | 11:35 | `SHORT` | 7528.70 | 7529.00 | -0.0040% | `LONG` | 4122.39 | 4131.19 | 0.2135% | **0.1047%** | 0.0262% | **0.2619%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 2** | 11:28 | 11:35 | `SHORT` | 7530.50 | 7529.00 | 0.0199% | `LONG` | 4122.29 | 4131.19 | 0.2159% | **0.1179%** | 0.0295% | **0.2948%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 3** | 11:28 | 11:35 | `SHORT` | 7530.50 | 7529.00 | 0.0199% | `LONG` | 4122.29 | 4131.19 | 0.2159% | **0.1179%** | 0.0295% | **0.2948%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 13:02 | 17:27 | `LONG` | 7522.60 | 7486.60 | -0.4786% | `SHORT` | 4135.52 | 4143.14 | -0.1843% | **-0.3314%** | -0.0829% | **-0.8285%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 2** | 13:18 | 17:27 | `LONG` | 7527.40 | 7486.60 | -0.5420% | `SHORT` | 4141.93 | 4143.14 | -0.0292% | **-0.2856%** | -0.0714% | **-0.7140%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 3** | 13:18 | 17:27 | `LONG` | 7527.40 | 7486.60 | -0.5420% | `SHORT` | 4141.93 | 4143.14 | -0.0292% | **-0.2856%** | -0.0714% | **-0.7140%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 4** | 13:35 | 17:27 | `LONG` | 7524.50 | 7486.60 | -0.5037% | `SHORT` | 4146.77 | 4143.14 | 0.0875% | **-0.2081%** | -0.0520% | **-0.5202%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 17:51 | 22:00 | `SHORT` | 7496.60 | 7500.80 | -0.0560% | `LONG` | 4141.34 | 4113.09 | -0.6821% | **-0.3691%** | -0.0923% | **-0.9227%** | `FORCED_EOD_CLOSE` |
| **Slot 2** | 17:54 | 22:00 | `SHORT` | 7496.30 | 7500.80 | -0.0600% | `LONG` | 4138.18 | 4113.09 | -0.6063% | **-0.3332%** | -0.0833% | **-0.8329%** | `FORCED_EOD_CLOSE` |
| **Slot 3** | 17:58 | 22:00 | `SHORT` | 7499.20 | 7500.80 | -0.0213% | `LONG` | 4137.78 | 4113.09 | -0.5967% | **-0.3090%** | -0.0773% | **-0.7725%** | `FORCED_EOD_CLOSE` |
