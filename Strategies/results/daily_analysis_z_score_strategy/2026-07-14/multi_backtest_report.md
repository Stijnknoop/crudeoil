# 📊 MANTRA: Layered Z-Score Session Report (2026-07-14)

* **Strategy Architecture:** `MULTI-SLOT GRID WITH VEILIGE ZONE BREAK-EVEN STOP`
* **Filters:** Expected Win (`>=0.15%`) | Dwell Block (`10m`) | Cluster Exit (`30m`) | BE Trigger (`|Z|=0.5`)

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 11
* **Batch Win Rate:** 36.36%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **-1.8639%**

### 📜 Session Transaction Ledger
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL Trade Combination | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 04:10 | 04:40 | `LONG` | 7506.30 | 7502.80 | `SHORT` | 4013.43 | 4015.29 | **-0.0465%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 04:11 | 04:40 | `LONG` | 7505.50 | 7502.80 | `SHORT` | 4012.91 | 4015.29 | **-0.0476%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 04:13 | 04:40 | `LONG` | 7501.90 | 7502.80 | `SHORT` | 4014.86 | 4015.29 | **0.0006%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 14:02 | 14:30 | `LONG` | 7507.10 | 7565.00 | `SHORT` | 4032.30 | 4089.53 | **-0.3240%** | `BREAK_EVEN_PROTECTION_EXIT` |
| **Slot 2** | 14:03 | 14:30 | `LONG` | 7506.80 | 7565.00 | `SHORT` | 4034.39 | 4089.53 | **-0.2957%** | `BREAK_EVEN_PROTECTION_EXIT` |
| **Slot 1** | 14:31 | 15:01 | `LONG` | 7558.70 | 7538.10 | `SHORT` | 4093.12 | 4083.01 | **-0.0128%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 14:32 | 15:01 | `LONG` | 7554.10 | 7538.10 | `SHORT` | 4094.17 | 4083.01 | **0.0304%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 14:32 | 15:01 | `LONG` | 7554.10 | 7538.10 | `SHORT` | 4094.17 | 4083.01 | **0.0304%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 4** | 14:32 | 15:01 | `LONG` | 7554.10 | 7538.10 | `SHORT` | 4094.17 | 4083.01 | **0.0304%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 17:59 | 18:29 | `SHORT` | 7536.20 | 7534.90 | `LONG` | 4066.48 | 4059.27 | **-0.0800%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 18:31 | 19:01 | `SHORT` | 7536.60 | 7547.00 | `LONG` | 4060.53 | 4063.64 | **-0.0307%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
