# 📊 MANTRA: Layered Z-Score Session Report (2026-07-07)

* **Strategy Architecture:** `PURE MATHEMATICAL MULTI-SLOT GRID`
* **Configured Slot Thresholds:** Slot 1 (`1.5`), Slot 2 (`2.0`), Slot 3 (`2.5`), Slot 4 (`3.0`)
* **Operational Windows:** Entries `04:00 - 20:00` | Forced Hard EOD Close `22:00`

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 12
* **Batch Win Rate:** 66.67%
* **Pure Combination Trade Yield (Rauw Totaal):** -0.2267%
* **Net Portfolio Session Yield (1x Base Portfolio):** -0.0567%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **-0.5668%**
* **Average Yield per Executed Slot (1x Base Portfolio):** -0.0047%
* **Average Yield per Executed Slot (10x Leveraged Portfolio):** -0.0472%

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
| **Slot 1** | 13:02 | 14:35 | `LONG` | 7522.60 | 7531.20 | 0.1143% | `SHORT` | 4135.52 | 4164.26 | -0.6950% | **-0.2903%** | -0.0726% | **-0.7258%** | `FORCED_EOD_CLOSE` |
| **Slot 2** | 13:18 | 14:35 | `LONG` | 7527.40 | 7531.20 | 0.0505% | `SHORT` | 4141.93 | 4164.26 | -0.5391% | **-0.2443%** | -0.0611% | **-0.6108%** | `FORCED_EOD_CLOSE` |
| **Slot 3** | 13:18 | 14:35 | `LONG` | 7527.40 | 7531.20 | 0.0505% | `SHORT` | 4141.93 | 4164.26 | -0.5391% | **-0.2443%** | -0.0611% | **-0.6108%** | `FORCED_EOD_CLOSE` |
| **Slot 4** | 13:35 | 14:35 | `LONG` | 7524.50 | 7531.20 | 0.0890% | `SHORT` | 4146.77 | 4164.26 | -0.4218% | **-0.1664%** | -0.0416% | **-0.4159%** | `FORCED_EOD_CLOSE` |
