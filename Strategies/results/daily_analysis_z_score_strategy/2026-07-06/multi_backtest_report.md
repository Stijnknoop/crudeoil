# 📊 MANTRA: Layered Z-Score Session Report (2026-07-06)

* **Strategy Architecture:** `PURE MATHEMATICAL MULTI-SLOT GRID WITH RISK CONTROLS`
* **Configured Slot Thresholds:** Slot 1 (`1.5`), Slot 2 (`2.0`), Slot 3 (`2.5`), Slot 4 (`3.0`)
* **Risk Configuration:** Max Z-Stop (`3.5`) | Freeze (`120m`) | Max Hold (`180m`)
* **Operational Windows:** Entries `04:00 - 20:00` | Forced Hard EOD Close `22:00`

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 8
* **Batch Win Rate:** 75.00%
* **Pure Combination Trade Yield (Rauw Totaal):** 0.2033%
* **Net Portfolio Session Yield (1x Base Portfolio):** 0.0508%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **0.5082%**
* **Average Yield per Executed Slot (1x Base Portfolio):** 0.0064%
* **Average Yield per Executed Slot (10x Leveraged Portfolio):** 0.0635%

### 📜 Session Transaction Ledger (Slot Decomposition)
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | PnL US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL GOLD | PnL Trade Combination | Cash PnL (1x) | Cash PnL (10x Leverage) | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 06:23 | 09:23 | `SHORT` | 7495.20 | 7504.70 | -0.1267% | `LONG` | 4161.95 | 4153.52 | -0.2025% | **-0.1646%** | -0.0412% | **-0.4116%** | `MAX_HOLDING_TIME_EXCEEDED` |
| **Slot 1** | 10:45 | 12:34 | `SHORT` | 7508.40 | 7512.80 | -0.0586% | `LONG` | 4149.47 | 4154.11 | 0.1118% | **0.0266%** | 0.0067% | **0.0665%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 2** | 11:29 | 12:34 | `SHORT` | 7510.10 | 7512.80 | -0.0360% | `LONG` | 4143.98 | 4154.11 | 0.2445% | **0.1042%** | 0.0261% | **0.2606%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 14:40 | 15:07 | `SHORT` | 7508.70 | 7506.60 | 0.0280% | `LONG` | 4140.28 | 4148.15 | 0.1901% | **0.1090%** | 0.0273% | **0.2726%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 15:57 | 18:37 | `SHORT` | 7514.30 | 7533.10 | -0.2502% | `LONG` | 4142.41 | 4148.87 | 0.1559% | **-0.0471%** | -0.0118% | **-0.1178%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 2** | 16:17 | 18:37 | `SHORT` | 7521.50 | 7533.10 | -0.1542% | `LONG` | 4138.95 | 4148.87 | 0.2397% | **0.0427%** | 0.0107% | **0.1068%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 19:32 | 21:23 | `LONG` | 7532.70 | 7550.30 | 0.2336% | `SHORT` | 4156.23 | 4162.12 | -0.1417% | **0.0460%** | 0.0115% | **0.1149%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 2** | 19:47 | 21:23 | `LONG` | 7530.70 | 7550.30 | 0.2603% | `SHORT` | 4158.49 | 4162.12 | -0.0873% | **0.0865%** | 0.0216% | **0.2162%** | `MEAN_REVERSION_CONVERGENCE` |
