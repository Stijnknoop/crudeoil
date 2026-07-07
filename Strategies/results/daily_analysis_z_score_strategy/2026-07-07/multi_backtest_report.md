# 📊 MANTRA: Layered Z-Score Session Report (2026-07-07)

* **Strategy Architecture:** `PURE MATHEMATICAL MULTI-SLOT GRID`
* **Configured Slot Thresholds:** Slot 1 (`1.5`), Slot 2 (`2.0`), Slot 3 (`2.5`), Slot 4 (`3.0`)
* **Risk Regulators:** Trailing Take Profit active (0.15%) | Emergency Portfolio Stop active (-0.25%)
* **Operational Windows (NL):** Entries `04:00 - 20:00` | Forced Hard EOD Close `22:00`

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 16
* **Batch Win Rate:** 50.00%
* **Pure Combination Trade Yield (Rauw Totaal):** -0.7975%
* **Net Portfolio Session Yield (1x Base Portfolio):** -0.1994%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **-1.9936%**
* **Average Yield per Executed Slot (1x Base Portfolio):** -0.0125%
* **Average Yield per Executed Slot (10x Leveraged Portfolio):** -0.1246%

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
| **Slot 1** | 13:02 | 14:09 | `LONG` | 7522.60 | 7532.50 | 0.1316% | `SHORT` | 4135.52 | 4169.72 | -0.8270% | **-0.3477%** | -0.0869% | **-0.8692%** | `PORTFOLIO_HARD_STOP_LOSS` |
| **Slot 2** | 13:18 | 14:09 | `LONG` | 7527.40 | 7532.50 | 0.0678% | `SHORT` | 4141.93 | 4169.72 | -0.6709% | **-0.3016%** | -0.0754% | **-0.7540%** | `PORTFOLIO_HARD_STOP_LOSS` |
| **Slot 3** | 13:18 | 14:09 | `LONG` | 7527.40 | 7532.50 | 0.0678% | `SHORT` | 4141.93 | 4169.72 | -0.6709% | **-0.3016%** | -0.0754% | **-0.7540%** | `PORTFOLIO_HARD_STOP_LOSS` |
| **Slot 4** | 13:35 | 14:09 | `LONG` | 7524.50 | 7532.50 | 0.1063% | `SHORT` | 4146.77 | 4169.72 | -0.5534% | **-0.2236%** | -0.0559% | **-0.5589%** | `PORTFOLIO_HARD_STOP_LOSS` |
| **Slot 1** | 14:10 | 17:05 | `LONG` | 7533.50 | 7489.90 | -0.5787% | `SHORT` | 4168.40 | 4150.88 | 0.4203% | **-0.0792%** | -0.0198% | **-0.1981%** | `FORCED_EOD_CLOSE` |
| **Slot 2** | 14:11 | 17:05 | `LONG` | 7533.20 | 7489.90 | -0.5748% | `SHORT` | 4166.28 | 4150.88 | 0.3696% | **-0.1026%** | -0.0256% | **-0.2564%** | `FORCED_EOD_CLOSE` |
| **Slot 3** | 14:11 | 17:05 | `LONG` | 7533.20 | 7489.90 | -0.5748% | `SHORT` | 4166.28 | 4150.88 | 0.3696% | **-0.1026%** | -0.0256% | **-0.2564%** | `FORCED_EOD_CLOSE` |
| **Slot 4** | 14:12 | 17:05 | `LONG` | 7533.10 | 7489.90 | -0.5735% | `SHORT` | 4170.02 | 4150.88 | 0.4590% | **-0.0572%** | -0.0143% | **-0.1431%** | `FORCED_EOD_CLOSE` |
