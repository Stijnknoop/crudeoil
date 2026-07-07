# 📊 MANTRA: Layered Z-Score Session Report (2026-07-07)

* **Strategy Architecture:** `PURE MATHEMATICAL MULTI-SLOT GRID WITH RISK REGULATORS`
* **Configured Slot Thresholds:** Slot 1 (`1.5`), Slot 2 (`2.0`), Slot 3 (`2.5`), Slot 4 (`3.0`)
* **Operational Guardrails:** Anti-Martingale Negative Block active | US Open Volatility Shield active
* **Operational Windows (NL):** Entries `04:00 - 20:00` | Forced Hard EOD Close `22:00`

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 4
* **Batch Win Rate:** 75.00%
* **Pure Combination Trade Yield (Rauw Totaal):** -0.1753%
* **Net Portfolio Session Yield (1x Base Portfolio):** -0.0438%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **-0.4383%**
* **Average Yield per Executed Slot (1x Base Portfolio):** -0.0110%
* **Average Yield per Executed Slot (10x Leveraged Portfolio):** -0.1096%

### 📜 Session Transaction Ledger (Slot Decomposition)
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | PnL US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL GOLD | PnL Trade Combination | Cash PnL (1x) | Cash PnL (10x Leverage) | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 04:05 | 06:08 | `SHORT` | 7534.90 | 7524.80 | 0.1340% | `LONG` | 4139.32 | 4137.46 | -0.0449% | **0.0446%** | 0.0111% | **0.1114%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 07:33 | 08:33 | `SHORT` | 7518.80 | 7521.00 | -0.0293% | `LONG` | 4121.50 | 4129.17 | 0.1861% | **0.0784%** | 0.0196% | **0.1960%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 10:53 | 11:35 | `SHORT` | 7528.70 | 7529.00 | -0.0040% | `LONG` | 4122.39 | 4131.19 | 0.2135% | **0.1047%** | 0.0262% | **0.2619%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 13:02 | 17:05 | `LONG` | 7522.60 | 7489.90 | -0.4347% | `SHORT` | 4135.52 | 4150.88 | -0.3714% | **-0.4031%** | -0.1008% | **-1.0076%** | `FORCED_EOD_CLOSE` |
