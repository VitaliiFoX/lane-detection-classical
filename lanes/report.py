"""Зведені таблиці й графіки для звіту. Задача M11, власник: Сивопляс Максим."""


def build_report(config: dict) -> None:
    """Читає ``results/metrics.csv`` і будує матеріали для звіту.

    - ``results/summary.csv`` — середні MSE, PSNR, SSIM, IoU і час для кожної
      пари (пайплайн, група) і рядок «усі групи»;
    - ``results/figures/`` — стовпчикові діаграми IoU, PSNR, SSIM і часу за
      групами (Matplotlib) і сітка прикладів «фото → стадії → результат».
    """
    raise NotImplementedError("M11")
