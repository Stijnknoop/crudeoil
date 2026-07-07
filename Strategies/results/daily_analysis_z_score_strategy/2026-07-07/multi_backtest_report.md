# 📊 MANTRA: Layered Z-Score Session Report (2026-07-07)

* **Strategy Architecture:** `PURE MATHEMATICAL MULTI-SLOT GRID WITH ACTIVE REGIME EXITS`
* **Configured Slot Thresholds:** Slot 1 (`1.5`), Slot 2 (`2.0`), Slot 3 (`2.5`), Slot 4 (`3.0`)
* **Regime Control Thresholds:** Max Slope (`±0.8%`) | Max Entry Dwell (`15m`) | Max Trade Holding (`30m`)
* **Operational Windows:** Entries `04:00 - 20:00` | Forced Hard EOD Close `22:00`

### 🔍 Trend vs. Mean-Reversion Regime Indicators
* **Max Ratio 240m Mean Slope (30m Delta):** `0.1377%`
* **Max Z-Score Dwell Time (|Z| >= 2.0):** `68 minutes`

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 3
* **Batch Win Rate:** 66.67%
* **Pure Combination Trade Yield (Rauw Totaal):** 0.1706%
* **Net Portfolio Session Yield (1x Base Portfolio):** 0.0427%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **0.4265%**
* **Average Yield per Executed Slot (1x Base Portfolio):** 0.0142%
* **Average Yield per Executed Slot (10x Leveraged Portfolio):** 0.1422%

### 📜 Session Transaction Ledger (Slot Decomposition)
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | PnL US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL GOLD | PnL Trade Combination | Cash PnL (1x) | Cash PnL (10x Leverage) | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 14:28 | 14:58 | `LONG` | 7533.80 | 7530.40 | -0.0451% | `SHORT` | 4168.41 | 4171.40 | -0.0717% | **-0.0584%** | -0.0146% | **-0.1461%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 15:36 | 16:06 | `LONG` | 7525.00 | 7516.50 | -0.1130% | `SHORT` | 4178.35 | 4155.41 | 0.5490% | **0.2180%** | 0.0545% | **0.5451%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 18:45 | 19:15 | `SHORT` | 7517.10 | 7515.30 | 0.0239% | `LONG` | 4142.37 | 4142.29 | -0.0019% | **0.0110%** | 0.0028% | **0.0275%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
