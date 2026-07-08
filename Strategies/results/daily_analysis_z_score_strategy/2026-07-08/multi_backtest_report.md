# 📊 MANTRA: Layered Z-Score Session Report (2026-07-08)

* **Strategy Architecture:** `PURE MATHEMATICAL MULTI-SLOT GRID WITH ACTIVE REGIME EXITS`
* **Configured Slot Thresholds:** Slot 1 (`1.5`), Slot 2 (`2.0`), Slot 3 (`2.5`), Slot 4 (`3.0`)
* **Regime Control Thresholds:** Max Slope (`±0.8%`) | Max Entry Dwell (`15m`) | Max Trade Holding (`30m`)
* **Operational Windows:** Entries `04:00 - 20:00` | Forced Hard EOD Close `22:00`

### 🔍 Trend vs. Mean-Reversion Regime Indicators
* **Max Ratio 240m Mean Slope (30m Delta):** `0.1057%`
* **Max Z-Score Dwell Time (|Z| >= 2.0):** `14 minutes`

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 2
* **Batch Win Rate:** 100.00%
* **Pure Combination Trade Yield (Rauw Totaal):** 0.1325%
* **Net Portfolio Session Yield (1x Base Portfolio):** 0.0331%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **0.3313%**
* **Average Yield per Executed Slot (1x Base Portfolio):** 0.0166%
* **Average Yield per Executed Slot (10x Leveraged Portfolio):** 0.1657%

### 📜 Session Transaction Ledger (Slot Decomposition)
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | PnL US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL GOLD | PnL Trade Combination | Cash PnL (1x) | Cash PnL (10x Leverage) | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 04:27 | 04:57 | `LONG` | 7500.80 | 7504.20 | 0.0453% | `SHORT` | 4122.03 | 4114.35 | 0.1863% | **0.1158%** | 0.0290% | **0.2896%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 05:01 | 05:31 | `LONG` | 7506.40 | 7511.20 | 0.0639% | `SHORT` | 4128.17 | 4129.43 | -0.0305% | **0.0167%** | 0.0042% | **0.0418%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
