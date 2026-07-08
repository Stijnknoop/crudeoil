# 📊 MANTRA: Layered Z-Score Session Report (2026-07-08)

* **Strategy Architecture:** `MULTI-SLOT GRID WITH VEILIGE ZONE BREAK-EVEN STOP`
* **Filters:** Expected Win (`>=0.1%`) | Dwell Block (`15m`) | Cluster Exit (`30m`) | BE Trigger (`|Z|=1.0`)

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 11
* **Batch Win Rate:** 18.18%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **-1.0933%**

### 📜 Session Transaction Ledger
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL Trade Combination | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 04:07 | 04:37 | `LONG` | 7500.50 | 7497.90 | `SHORT` | 4114.92 | 4119.06 | **-0.0676%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 04:10 | 04:37 | `LONG` | 7500.60 | 7497.90 | `SHORT` | 4117.11 | 4119.06 | **-0.0417%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 04:38 | 05:01 | `LONG` | 7498.60 | 7505.80 | `SHORT` | 4120.30 | 4128.47 | **-0.0511%** | `BREAK_EVEN_PROTECTION_EXIT` |
| **Slot 1** | 05:02 | 05:32 | `LONG` | 7506.50 | 7509.30 | `SHORT` | 4125.32 | 4128.50 | **-0.0199%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 08:12 | 08:42 | `LONG` | 7492.30 | 7486.30 | `SHORT` | 4129.59 | 4126.91 | **-0.0076%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 08:14 | 08:42 | `LONG` | 7492.80 | 7486.30 | `SHORT` | 4131.55 | 4126.91 | **0.0128%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 10:16 | 10:46 | `SHORT` | 7475.70 | 7438.90 | `LONG` | 4106.03 | 4069.45 | **-0.1993%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 10:17 | 10:46 | `SHORT` | 7468.00 | 7438.90 | `LONG` | 4087.55 | 4069.45 | **-0.0266%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 10:17 | 10:46 | `SHORT` | 7468.00 | 7438.90 | `LONG` | 4087.55 | 4069.45 | **-0.0266%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 4** | 10:17 | 10:46 | `SHORT` | 7468.00 | 7438.90 | `LONG` | 4087.55 | 4069.45 | **-0.0266%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 11:43 | 12:04 | `SHORT` | 7425.20 | 7424.20 | `LONG` | 4045.73 | 4046.55 | **0.0169%** | `FORCED_EOD_CLOSE` |
