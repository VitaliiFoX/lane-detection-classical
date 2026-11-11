# lanes

Пакет програми. Шари й контракти даних — `ARCHITECTURE.md`.

| Модуль | Що робить | Задача | Власник |
|---|---|---|---|
| `config.py` | читання `config.json` | готово | Фояк |
| `io_utils.py` | фото, розмітка `.lines.txt`, PNG | F2 | Фояк |
| `filters.py` | `apply_kernel` — ядро 3×3/5×5 | F1 | Фояк |
| `draw.py` | маски смуг, фото з лініями | F4 | Фояк |
| `metrics.py` | MSE, PSNR, SSIM, IoU | F5 | Фояк |
| `runner.py` | обробка вибірки, `metrics.csv` | F11 | Фояк |
| `roi.py` | трапеція області інтересу | M1 | Максим |
| `synthetic.py` | синтетичний кадр із відомими смугами | M2 | Максим |
| `checks.py` | відхилення смуг, сценарії C1 | M10 | Максим |
| `report.py` | `summary.csv`, графіки | M11 | Максим |
| `methods/` | методи обробки | — | див. `methods/README.md` |
| `pipelines/` | пайплайни p0–p4 | — | див. `pipelines/README.md` |
