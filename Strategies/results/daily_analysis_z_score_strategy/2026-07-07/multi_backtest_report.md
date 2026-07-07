# 📊 MANTRA: Layered Z-Score Session Report (2026-07-07)

* **Strategy Architecture:** `PURE MATHEMATICAL MULTI-SLOT GRID WITH ACTIVE REGIME EXITS`
* **Configured Slot Thresholds:** Slot 1 (`1.5`), Slot 2 (`2.0`), Slot 3 (`2.5`), Slot 4 (`3.0`)
* **Regime Control Thresholds:** Max Slope (`±0.08%`) | Max Entry Dwell (`15m`) | Max Trade Holding (`30m`)
* **Operational Windows:** Entries `04:00 - 20:00` | Forced Hard EOD Close `22:00`

### 🔍 Trend vs. Mean-Reversion Regime Indicators
* **Max Ratio 240m Mean Slope (30m Delta):** `0.1377%`
* **Max Z-Score Dwell Time (|Z| >= 2.0):** `68 minutes`

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 1
* **Batch Win Rate:** 100.00%
* **Pure Combination Trade Yield (Rauw Totaal):** 0.0110%
* **Net Portfolio Session Yield (1x Base Portfolio):** 0.0028%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **0.0275%**
* **Average Yield per Executed Slot (1x Base Portfolio):** 0.0028%
* **Average Yield per Executed Slot (10x Leveraged Portfolio):** 0.0275%

### 📜 Session Transaction Ledger (Slot Decomposition)
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | PnL US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL GOLD | PnL Trade Combination | Cash PnL (1x) | Cash PnL (10x Leverage) | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 18:45 | 19:15 | `SHORT` | 7517.10 | 7515.30 | 0.0239% | `LONG` | 4142.37 | 4142.29 | -0.0019% | **0.0110%** | 0.0028% | **0.0275%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
