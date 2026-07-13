# 📊 MANTRA: Layered Z-Score Session Report (2026-07-13)

* **Strategy Architecture:** `MULTI-SLOT GRID WITH VEILIGE ZONE BREAK-EVEN STOP`
* **Filters:** Expected Win (`>=0.15%`) | Dwell Block (`10m`) | Cluster Exit (`30m`) | BE Trigger (`|Z|=0.5`)

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 5
* **Batch Win Rate:** 100.00%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **0.6865%**

### 📜 Session Transaction Ledger
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL Trade Combination | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 05:11 | 05:41 | `SHORT` | 7538.30 | 7540.90 | `LONG` | 4055.41 | 4058.99 | **0.0269%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 05:12 | 05:41 | `SHORT` | 7537.30 | 7540.90 | `LONG` | 4054.95 | 4058.99 | **0.0259%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 08:06 | 08:29 | `SHORT` | 7531.20 | 7537.20 | `LONG` | 4044.60 | 4061.06 | **0.1636%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 10:21 | 10:44 | `LONG` | 7551.60 | 7550.50 | `SHORT` | 4078.25 | 4075.89 | **0.0217%** | `FORCED_EOD_CLOSE` |
| **Slot 2** | 10:22 | 10:44 | `LONG` | 7550.60 | 7550.50 | `SHORT` | 4078.92 | 4075.89 | **0.0365%** | `FORCED_EOD_CLOSE` |
