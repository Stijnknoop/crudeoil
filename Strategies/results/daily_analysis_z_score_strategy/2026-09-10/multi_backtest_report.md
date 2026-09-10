# 📊 MANTRA: Layered Z-Score Session Report (2026-09-10)

* **Strategy Architecture:** `MULTI-SLOT GRID WITH VEILIGE ZONE BREAK-EVEN STOP`
* **Filters:** Expected Win (`>=0.15%`) | Dwell Block (`10m`) | Cluster Exit (`30m`) | BE Trigger (`|Z|=0.5`)

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 8
* **Batch Win Rate:** 75.00%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **0.6298%**

### 📜 Session Transaction Ledger
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL Trade Combination | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 07:59 | 08:14 | `LONG` | 7657.30 | 7655.00 | `SHORT` | 4430.81 | 4416.76 | **0.1435%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 10:45 | 11:15 | `SHORT` | 7647.20 | 7646.30 | `LONG` | 4394.15 | 4393.78 | **0.0017%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 12:36 | 13:06 | `SHORT` | 7643.50 | 7638.10 | `LONG` | 4380.09 | 4379.10 | **0.0240%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 14:31 | 14:54 | `SHORT` | 7611.60 | 7599.80 | `LONG` | 4353.15 | 4343.17 | **-0.0371%** | `BREAK_EVEN_PROTECTION_EXIT` |
| **Slot 2** | 14:32 | 14:54 | `SHORT` | 7608.70 | 7599.80 | `LONG` | 4351.47 | 4343.17 | **-0.0369%** | `BREAK_EVEN_PROTECTION_EXIT` |
| **Slot 3** | 14:34 | 14:54 | `SHORT` | 7598.80 | 7599.80 | `LONG` | 4342.58 | 4343.17 | **0.0002%** | `BREAK_EVEN_PROTECTION_EXIT` |
| **Slot 4** | 14:35 | 14:54 | `SHORT` | 7603.30 | 7599.80 | `LONG` | 4341.24 | 4343.17 | **0.0452%** | `BREAK_EVEN_PROTECTION_EXIT` |
| **Slot 1** | 15:48 | 16:18 | `LONG` | 7597.50 | 7589.20 | `SHORT` | 4374.35 | 4359.84 | **0.1112%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
