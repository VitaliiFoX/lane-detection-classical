# Виявлення смуг дорожньої розмітки класичними методами

STEM-проєкт з дисципліни «Комп'ютерна графіка та візуалізація» (КНУ, ФІТ).
Команда: **Фояк Віталій**, **Сивопляс Максим**.

Програма знаходить смуги розмітки на фото з камери автомобіля чотирма
пайплайнами класичних методів комп'ютерної графіки й порівнює їх на 100 фото
відкритого набору [CULane](https://xingangpan.github.io/projects/CULane.html).
Усі алгоритми написано вручну: без OpenCV, без ШІ та машинного навчання.

## Статус

| Частина | Стан |
|---|---|
| Каркас, документація, план | ✅ |
| Спільна інфраструктура (читання, маски, метрики, ROI, раннер) | ✅ |
| p1 «Колір», p2 «Оцу» — Фояк | ✅ |
| p3 «Вид зверху», p4 «Top-hat» — Сивопляс | ✅ |
| Вибірка 100 фото | 🚧 |
| Експерименти та звіт | 🚧 |

## Пайплайни

| № | Стадії | Автор |
|---|---|---|
| p1 | HSV-маска білого й жовтого → фільтр Собеля → перетворення Хафа | Фояк Віталій |
| p2 | вирівнювання гістограми → поріг Оцу → перетворення Хафа | Фояк Віталій |
| p3 | вид зверху → фільтр Прюітта → гістограма стовпців із ковзними вікнами | Сивопляс Максим |
| p4 | морфологічний top-hat → метод Кенні → перетворення Хафа | Сивопляс Максим |

Подробиці — [docs/pipelines.md](docs/pipelines.md).

## Запуск з нуля

Потрібен Python 3.10 або новіший.

```bash
git clone <посилання на репозиторій>
cd lane-detection-classical
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest
```

Вибірка в репозиторій не входить (ліцензія CULane). Як її отримати —
[docs/data.md](docs/data.md). Коли фото лежать у `data/sample/`:

```bash
python main.py run --pipeline all     # обробка, проміжні стадії, metrics.csv
python main.py run --pipeline p3 --limit 10
python main.py report                 # summary.csv і графіки
python main.py check                  # еталонні сценарії
```

Усі параметри методів — у [config.json](config.json). Результати — у
`results/`: для кожного фото проміжні стадії (`01_…png`), маска й фото з
лініями, а також `metrics.csv`, `summary.csv`, `c1_check.csv`, `figures/`.

## Структура

```
main.py              точка входу: run, report, check
config.json          усі параметри
lanes/               спільна інфраструктура: читання, ROI, маски, метрики, раннер
lanes/methods/       методи обробки, по одному файлу на метод
lanes/pipelines/     пайплайни p0–p4
tests/               автотести
tools/make_sample.py збирає вибірку з CULane
docs/                специфікації, дані, план звіту
```

## Документація

| Документ | Про що |
|---|---|
| [PLAN.md](PLAN.md) | задачі, власники, залежності |
| [docs/pipelines.md](docs/pipelines.md) | специфікація пайплайнів і параметрів |
| [docs/metrics.md](docs/metrics.md) | метрики, еталонні сценарії, пороги |
| [docs/data.md](docs/data.md) | як завантажити CULane і зібрати вибірку |
| [docs/report-outline.md](docs/report-outline.md) | план звіту і хто що пише |
| [ARCHITECTURE.md](ARCHITECTURE.md) | модулі й контракти даних |
| [ADR.md](ADR.md) | ключові рішення і чому саме так |
| [AGENTS.md](AGENTS.md) | правила роботи для ШІ-асистентів |
| [META-USAGE.md](META-USAGE.md) | декларація використання ШІ |

`AGENTS.md` читають Claude Code, Codex і Antigravity; `CLAUDE.md` імпортує
його. Для Claude Code є скіли `/next-task`, `/check`, `/defense-prep`.
