# 📊 MANTRA: Layered Z-Score Session Report (2026-07-08)

* **Strategy Architecture:** `MULTI-SLOT GRID WITH VEILIGE ZONE BREAK-EVEN STOP`
* **Filters:** Expected Win (`>=0.15%`) | Dwell Block (`15m`) | Cluster Exit (`30m`) | BE Trigger (`|Z|=1.5`)

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 12
* **Batch Win Rate:** 16.67%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **-1.0925%**

### 📜 Session Transaction Ledger
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL Trade Combination | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 04:10 | 04:40 | `LONG` | 7500.60 | 7499.30 | `SHORT` | 4117.11 | 4119.41 | **-0.0366%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 04:22 | 04:40 | `LONG` | 7499.90 | 7499.30 | `SHORT` | 4118.69 | 4119.41 | **-0.0127%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 05:01 | 05:31 | `LONG` | 7506.40 | 7511.20 | `SHORT` | 4128.17 | 4129.43 | **0.0167%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 10:17 | 10:40 | `SHORT` | 7468.00 | 7432.10 | `LONG` | 4087.55 | 4064.90 | **-0.0367%** | `BREAK_EVEN_PROTECTION_EXIT` |
| **Slot 2** | 10:18 | 10:40 | `SHORT` | 7473.00 | 7432.10 | `LONG` | 4087.66 | 4064.90 | **-0.0047%** | `BREAK_EVEN_PROTECTION_EXIT` |
| **Slot 3** | 10:18 | 10:40 | `SHORT` | 7473.00 | 7432.10 | `LONG` | 4087.66 | 4064.90 | **-0.0047%** | `BREAK_EVEN_PROTECTION_EXIT` |
| **Slot 4** | 10:18 | 10:40 | `SHORT` | 7473.00 | 7432.10 | `LONG` | 4087.66 | 4064.90 | **-0.0047%** | `BREAK_EVEN_PROTECTION_EXIT` |
| **Slot 1** | 10:41 | 11:11 | `SHORT` | 7429.30 | 7426.10 | `LONG` | 4062.03 | 4052.68 | **-0.0936%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 10:42 | 11:11 | `SHORT` | 7433.20 | 7426.10 | `LONG` | 4064.06 | 4052.68 | **-0.0922%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 10:42 | 11:11 | `SHORT` | 7433.20 | 7426.10 | `LONG` | 4064.06 | 4052.68 | **-0.0922%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 4** | 10:42 | 11:11 | `SHORT` | 7433.20 | 7426.10 | `LONG` | 4064.06 | 4052.68 | **-0.0922%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 11:43 | 12:04 | `SHORT` | 7425.20 | 7424.20 | `LONG` | 4045.73 | 4046.55 | **0.0169%** | `FORCED_EOD_CLOSE` |
