# Декларація використання ШІ-інструментів

За політикою щодо генеративного ШІ з методички використання ШІ дозволене за
трьох умов: (1) воно задеклароване тут; (2) автор володіє матеріалом і може
порівняти обрану стратегію з альтернативами; (3) автор розуміє згенерований
код і може пояснити його коректність. Умови (2) і (3) перевіряються на захисті.

Новий рядок — після кожної задачі з `PLAN.md`, у якій працював ШІ.

| Дата | Хто | Інструмент | Задача | Ключовий запит | Що згенеровано | Що перевірив або змінив автор |
|---|---|---|---|---|---|---|
| 2026-11-11 | Фояк Віталій | Claude Code (Claude Opus 5.5) | каркас | «оформи проєкт у форматі cryptarc: план, пайплайни, правила, щоб асистенти могли працювати за планом» | структура репозиторію, README, AGENTS.md, PLAN.md, ARCHITECTURE.md, ADR.md, docs/, config.json, заготовки модулів із сигнатурами, скіли `.claude/skills` | склад пайплайнів і пороги — з погодженого з викладачкою ТЗ |
| 2026-11-13 | Фояк Віталій | Claude Code (Claude Opus 5.5) | F1 | «виконуй мої таски» | `lanes/filters.py`, `tests/test_filters.py` | потребує перевірки автором |
| 2026-11-13 | Фояк Віталій | Claude Code (Claude Opus 5.5) | F2 | «виконуй мої таски» | `lanes/io_utils.py`, `tests/test_io_utils.py` | потребує перевірки автором |
| 2026-11-13 | Фояк Віталій | Claude Code (Claude Opus 5.5) | F4 | «виконуй мої таски» | `lanes/draw.py`, `tests/test_draw.py` | потребує перевірки автором |
| 2026-11-13 | Фояк Віталій | Claude Code (Claude Opus 5.5) | F5 | «виконуй мої таски» | `lanes/metrics.py`, `tests/test_metrics.py` | потребує перевірки автором |
| 2026-11-13 | Фояк Віталій | Claude Code (Claude Opus 5.5) | F6 | «виконуй мої таски» | `lanes/methods/color.py`, `tests/test_color.py` | потребує перевірки автором |
| 2026-11-13 | Фояк Віталій | Claude Code (Claude Opus 5.5) | F7 | «виконуй мої таски» | `lanes/methods/sobel.py`, `tests/test_sobel.py` | потребує перевірки автором |
| 2026-11-13 | Фояк Віталій | Claude Code (Claude Opus 5.5) | F8 | «виконуй мої таски» | `lanes/methods/hough.py`, `tests/test_hough.py`, уточнення в `docs/pipelines.md` | потребує перевірки автором |
| 2026-11-13 | Фояк Віталій | Claude Code (Claude Opus 5.5) | F9 | «виконуй мої таски» | `lanes/methods/histogram.py`, `tests/test_histogram.py` | потребує перевірки автором |
| 2026-11-13 | Фояк Віталій | Claude Code (Claude Opus 5.5) | F10 | «виконуй мої таски» | `lanes/methods/threshold.py`, `tests/test_threshold.py` | потребує перевірки автором |
| 2026-11-13 | Фояк Віталій | Claude Code (Claude Opus 5.5) | F11 | «виконуй мої таски» | `lanes/runner.py`, `lanes/config.py` (`project_path`), `main.py`, `tests/test_runner.py` | потребує перевірки автором |
| 2026-11-13 | Фояк Віталій | Claude Code (Claude Opus 5.5) | F3 (код, без даних) | «виконуй мої таски» | `tools/make_sample.py`, `tests/test_make_sample.py` | потребує перевірки автором; задача відкрита, доки немає вибірки |
| 2026-11-16 | Сивопляс Максим | Claude Code (Claude Fable 5.2) | M1 | «знайди в плані мої таски і виконай їх» | `lanes/roi.py`, `tests/test_roi.py` | потребує перевірки автором |
| 2026-11-16 | Сивопляс Максим | Claude Code (Claude Fable 5.2) | M2 | «знайди в плані мої таски і виконай їх» | `lanes/synthetic.py`, `tests/test_synthetic.py` | потребує перевірки автором |
| 2026-11-16 | Сивопляс Максим | Claude Code (Claude Fable 5.2) | M3 | «знайди в плані мої таски і виконай їх» | `lanes/methods/perspective.py` (додано допоміжну `top_view_rectangle`), `tests/test_perspective.py` | потребує перевірки автором |
| 2026-11-16 | Сивопляс Максим | Claude Code (Claude Fable 5.2) | M4 | «знайди в плані мої таски і виконай їх» | `lanes/methods/morphology.py`, `tests/test_morphology.py` | потребує перевірки автором |
| 2026-11-16 | Сивопляс Максим | Claude Code (Claude Fable 5.2) | M5 | «знайди в плані мої таски і виконай їх» | `lanes/methods/prewitt.py`, `tests/test_prewitt.py` | потребує перевірки автором |
| 2026-11-16 | Сивопляс Максим | Claude Code (Claude Fable 5.2) | M6 | «знайди в плані мої таски і виконай їх» | `lanes/methods/sliding_windows.py`, `tests/test_sliding_windows.py` | потребує перевірки автором |
| 2026-11-16 | Сивопляс Максим | Claude Code (Claude Fable 5.2) | M7 | «знайди в плані мої таски і виконай їх» | `lanes/methods/canny.py`, `tests/test_canny.py` | потребує перевірки автором |
| 2026-11-16 | Сивопляс Максим | Claude Code (Claude Fable 5.2) | M10 | «знайди в плані мої таски і виконай їх» | `lanes/checks.py`, `tests/test_checks.py`, ключ `checks.offset_levels` у `config.json`, уточнення в `docs/metrics.md` | потребує перевірки автором |
| 2026-11-16 | Сивопляс Максим | Claude Code (Claude Fable 5.2) | M8 | «знайди в плані мої таски і виконай їх» | `lanes/pipelines/p3_top_view.py`, `tests/test_p3_top_view.py`, `checks.synthetic_offset` для тестів пайплайнів, уточнення в `docs/pipelines.md` (маска фото) | потребує перевірки автором |
| 2026-11-16 | Сивопляс Максим | Claude Code (Claude Fable 5.2) | M9 | «знайди в плані мої таски і виконай їх» | `lanes/pipelines/p4_tophat.py`, `tests/test_p4_tophat.py` | потребує перевірки автором |
| 2026-11-16 | Сивопляс Максим | Claude Code (Claude Fable 5.2) | M12 (частина: p0, p3, p4) | «знайди в плані мої таски і виконай їх» | `tests/test_edge_cases.py` | потребує перевірки автором; задача відкрита, доки немає p1 і p2 (F12, F13) |
| 2026-11-16 | Сивопляс Максим | Claude Code (Claude Fable 5.2) | M11 | «знайди в плані мої таски і виконай їх» | `lanes/report.py`, `tests/test_report.py`, уточнення в `docs/metrics.md` (формат `summary.csv`, назви графіків) | потребує перевірки автором |
