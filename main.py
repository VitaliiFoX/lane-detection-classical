"""Точка входу: ``python main.py run | report | check``. Власник: Фояк Віталій.

Тут лише розбір аргументів: уся робота — у ``lanes/runner.py``,
``lanes/report.py`` і ``lanes/checks.py``.
"""

import argparse

from lanes import checks, report, runner
from lanes.config import DEFAULT_CONFIG, load_config
from lanes.pipelines import REGISTRY


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Розбирає аргументи командного рядка."""
    parser = argparse.ArgumentParser(
        description="Виявлення смуг дорожньої розмітки класичними методами.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="шлях до файлу параметрів (за замовчуванням config.json)")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="обробити вибірку і порахувати метрики")
    run_parser.add_argument("--pipeline", default="all", choices=["all", *REGISTRY],
                            help="який пайплайн запустити (за замовчуванням усі)")
    run_parser.add_argument("--limit", type=int, default=None,
                            help="обробити лише перші N фото вибірки")

    commands.add_parser("report", help="summary.csv і графіки з metrics.csv")
    commands.add_parser("check", help="еталонні сценарії (критерій C1)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Запускає обрану команду з параметрами з конфігураційного файлу."""
    args = parse_args(argv)
    config = load_config(args.config)
    try:
        if args.command == "run":
            names = list(REGISTRY) if args.pipeline == "all" else [args.pipeline]
            runner.run_pipelines(config, names, args.limit)
        elif args.command == "report":
            report.build_report(config)
        else:
            checks.run_checks(config)
    except FileNotFoundError as error:
        # Немає вибірки чи результатів — коротке повідомлення замість трасування.
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
