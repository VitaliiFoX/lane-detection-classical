"""Вікно, що показує, як кожен метод крок за кроком знаходить смуги розмітки. Власник: Фояк Віталій.

Запуск: ``python gui.py``. Обробка — ті самі функції, що й у ``main.py run``;
вікно лише показує проміжні кроки простими словами, порівнює чотири методи на
одному фото й дає змінити параметри, не чіпаючи ``config.json``.
"""

import copy
import json
import time
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from lanes.config import load_config, project_path
from lanes.draw import lanes_mask, overlay
from lanes.io_utils import load_rgb, read_lanes, to_uint8
from lanes.metrics import iou, psnr, ssim
from lanes.pipelines import REGISTRY
from lanes.roi import roi_polygon
from lanes.runner import lines_path, read_sample

PICTURE_SIZE = (960, 400)  # найбільший розмір великої картинки, px
SMALL_SIZE = (520, 200)    # картинки на вкладці порівняння
ROI_COLOR = (255, 220, 0)
COMPARED = ["p1", "p2", "p3", "p4"]

GROUPS = {"normal": "Звичайна дорога", "crowd": "Затор", "shadow": "Тінь",
          "curve": "Поворот", "night": "Ніч"}

# Ідея кожного методу одним реченням.
METHODS = {
    "p0": "Нічого не шукає. Потрібен лише для порівняння: які оцінки отримує порожня відповідь.",
    "p1": "Шукає білу й жовту фарбу, потім проводить прямі через межі знайдених плям.",
    "p2": "Підсилює контраст і відділяє найсвітліші місця, потім проводить через них прямі.",
    "p3": "«Розгортає» дорогу, ніби дивимось з дрона, і йде віконцями вздовж кожної смуги.",
    "p4": "Лишає тільки вузькі світлі лінії, знаходить їхні тонкі межі й проводить прямі.",
}

# Крок: назва простими словами й пояснення — спершу що видно, потім технічна назва.
STEPS = {
    "original": ("Фото", "Кадр з камери автомобіля — так його отримує програма."),
    "roi": ("Зона пошуку",
            "Смуги шукаємо лише в жовтій зоні — це дорога перед машиною. Небо, дерева й "
            "капот відкидаємо, щоб вони не заважали. Технічно: область інтересу (ROI)."),
    "hsv_mask": ("Біла й жовта фарба",
                 "Білим позначено пікселі, схожі кольором на фарбу розмітки: світлі й майже "
                 "без кольору або жовті. Технічно: переводимо RGB у HSV і ставимо пороги на "
                 "яскравість V, насиченість S і відтінок H."),
    "sobel_magnitude": ("Межі плям",
                        "Лишаємо лише краї білих плям — там, де яскравість різко змінюється. "
                        "Технічно: фільтр Собеля, модуль градієнта."),
    "edges_roi": ("Межі в зоні пошуку",
                  "Найсильніші межі, і лише всередині зони пошуку. Саме через ці точки "
                  "шукатимемо прямі."),
    "hough_space": ("Голосування за прямі",
                    "Кожна точка «голосує» за всі прямі, що через неї проходять. Яскраві "
                    "плями — прямі, за які проголосувало найбільше точок; їх і беремо. "
                    "Технічно: перетворення Хафа, осі — кут θ і відстань ρ."),
    "equalized": ("Підсилений контраст",
                  "Фото стає сірим, а яскравості розтягуються на весь діапазон: світле — "
                  "світлішим, темне — темнішим. Технічно: вирівнювання гістограми."),
    "otsu": ("Світле окремо від темного",
             "Програма сама обирає поріг яскравості, що найкраще ділить зону пошуку на світле "
             "й темне (двічі поспіль). Біле — найсвітліші місця, серед них має бути "
             "розмітка. Технічно: метод Оцу."),
    "top_view": ("Вид зверху",
                 "Дорогу «розгортаємо», ніби дивимось на неї з дрона: смуги стають майже "
                 "вертикальними паралельними лініями. Технічно: перспективне перетворення."),
    "prewitt_magnitude": ("Межі на виді зверху",
                          "Знаходимо краї — місця, де яскравість різко змінюється. "
                          "Технічно: фільтр Прюітта."),
    "edges": ("Найсильніші межі", "Лишаємо лише найсильніші краї — кандидатів у розмітку."),
    "windows": ("Пошук віконцями",
                "Внизу знаходимо стовпці з найбільшою кількістю точок — там починаються смуги. "
                "Віконце піднімається вздовж кожної смуги, а через точки у вікнах проводимо "
                "криву (параболу)."),
    "tophat": ("Тонкі світлі лінії",
               "Лишаються тільки вузькі світлі деталі, як-от розмітка; широкі світлі ділянки "
               "(асфальт, небо) зникають. Технічно: морфологічний top-hat."),
    "canny": ("Тонкі межі",
              "Знаходимо тонкі, в один піксель, неперервні межі. Технічно: метод Кенні."),
    "mask": ("Знайдене (маска)",
             "Знайдені лінії, намальовані білим. Саме цю картинку порівнюємо з правильною "
             "розміткою, щоб порахувати оцінки праворуч."),
    "result": ("Результат",
               "Червоне — що знайшов метод. Зелене — де смуги насправді (розмітка набору "
               "CULane). Що ближче червоне до зеленого, то краще."),
}



def draw_roi(image: np.ndarray, config: dict) -> np.ndarray:
    """Копія фото з контуром зони пошуку (жовтий многокутник)."""
    canvas = Image.fromarray(np.array(image, dtype=np.uint8))
    points = roi_polygon(image.shape[:2], config["roi"]["polygon"])
    ImageDraw.Draw(canvas).line(points + points[:1], fill=ROI_COLOR, width=6)
    return np.array(canvas)


def process(image: np.ndarray, gt_lanes: list[np.ndarray], name: str,
            config: dict) -> tuple[list[tuple[str, np.ndarray, str]], dict]:
    """Запускає метод ``name`` і повертає (кроки для показу, оцінки).

    Крок — ``(назва, масив, пояснення)``: фото, зона пошуку, усі стадії
    пайплайна, маска знайденого і результат поверх фото. Оцінки рахуються
    так само, як у ``runner``.
    """
    start = time.perf_counter()
    result = REGISTRY[name].run(image, config)
    elapsed_ms = (time.perf_counter() - start) * 1000

    thickness = config["masks"]["thickness"]
    found = lanes_mask(image.shape[:2], result.lanes, thickness)
    expected = lanes_mask(image.shape[:2], gt_lanes, thickness)

    def step(key: str, array: np.ndarray) -> tuple[str, np.ndarray, str]:
        title, text = STEPS.get(key, (key, ""))
        return title, array, text

    steps = [step("original", image), step("roi", draw_roi(image, config))]
    steps += [step(stage, array) for stage, array in result.stages]
    steps += [step("mask", found), step("result", overlay(image, result.lanes, gt_lanes))]

    settings = config["metrics"]
    scores = {
        "iou": iou(expected, found),
        "psnr": psnr(expected, found, settings["psnr_cap"]),
        "ssim": ssim(expected, found, settings["ssim_window"], settings["ssim_k1"],
                     settings["ssim_k2"]),
        "found": len(result.lanes),
        "expected": len(gt_lanes),
        "time_ms": elapsed_ms,
    }
    return steps, scores


def number(value: float, digits: int = 2) -> str:
    """Число з комою, як прийнято в українському тексті: 0.45 → «0,45»."""
    return f"{value:.{digits}f}".replace(".", ",")


def to_photo(array: np.ndarray, size: tuple[int, int]) -> ImageTk.PhotoImage:
    """Масив → картинка Tk. Зменшення лише для показу; в обробці Pillow не бере участі."""
    picture = Image.fromarray(to_uint8(array))
    picture.thumbnail(size)
    return ImageTk.PhotoImage(picture)


class App:  # pylint: disable=too-many-instance-attributes
    """Головне вікно: вибір фото й методу, кроки методу, порівняння, оцінки, налаштування."""

    def __init__(self, root: tk.Tk, config: dict):
        self.root = root
        self.config = config
        self.samples = read_sample(project_path(config["data"]["sample_csv"]))
        self.sample_dir = project_path(config["data"]["sample_dir"])
        self.photo_index = 0
        self.method = tk.StringVar(value="p1")
        self.step = tk.IntVar(value=0)
        self.steps: list[tuple[str, np.ndarray, str]] = []
        self.pictures: dict[str, ImageTk.PhotoImage] = {}  # інакше Tk видалить картинки
        self.entries: dict[tuple[str, str], ttk.Entry] = {}
        self.image = None
        self.gt_lanes = []

        root.title("Пошук смуг дорожньої розмітки")
        self._build_header()
        self._build_choice()
        body = ttk.Frame(root, padding=(12, 0, 12, 12))
        body.pack(fill="both", expand=True)
        self._build_tabs(body)
        self._build_side(body)
        root.bind("<Left>", lambda _event: self.move_step(-1))
        root.bind("<Right>", lambda _event: self.move_step(1))
        self.on_method()
        self.load_photo()

    # ------------------------------------------------------------ віджети
    def _build_header(self):
        header = ttk.Frame(self.root, padding=(12, 10, 12, 0))
        header.pack(fill="x")
        ttk.Label(header, text="Пошук смуг дорожньої розмітки на фото",
                  font=("Helvetica", 20, "bold")).pack(anchor="w")
        ttk.Label(header, wraplength=1350, font=("Helvetica", 13), text=(
            "Програма знаходить смуги розмітки на фото з камери автомобіля чотирма методами "
            "класичної комп'ютерної графіки (без нейромереж). Оберіть фото й метод. Вкладка "
            "«Як працює метод» показує кроки по черзі — кнопки «Назад» / «Далі» або стрілки "
            "← →. Вкладка «Порівняння методів» показує результат усіх чотирьох методів на "
            "цьому фото.")).pack(anchor="w")

    def _build_choice(self):
        choice = ttk.Frame(self.root, padding=(12, 10))
        choice.pack(fill="x")
        bold = ("Helvetica", 14, "bold")

        row = ttk.Frame(choice)
        row.pack(fill="x")
        ttk.Label(row, text="1. Фото:", font=bold, width=10).pack(side="left")
        self.group = ttk.Combobox(row, state="readonly", width=18,
                                  values=[GROUPS.get(group, group) for group in self.groups()])
        self.group.current(0)
        self.group.bind("<<ComboboxSelected>>", lambda _event: self.on_group())
        self.group.pack(side="left")
        ttk.Button(row, text="◀", width=3, command=lambda: self.move_photo(-1)).pack(
            side="left", padx=(12, 0))
        self.photo_label = ttk.Label(row, width=14, anchor="center")
        self.photo_label.pack(side="left")
        ttk.Button(row, text="▶", width=3, command=lambda: self.move_photo(1)).pack(side="left")
        self.file_label = ttk.Label(row, foreground="gray")
        self.file_label.pack(side="left", padx=12)

        row = ttk.Frame(choice)
        row.pack(fill="x", pady=(8, 0))
        ttk.Label(row, text="2. Метод:", font=bold, width=10).pack(side="left")
        for name in COMPARED + ["p0"]:
            ttk.Radiobutton(row, text=f"{name} «{REGISTRY[name].title}»", value=name,
                            variable=self.method, command=self.on_method).pack(side="left",
                                                                                padx=(0, 14))
        self.method_label = ttk.Label(choice, font=("Helvetica", 13), foreground="gray")
        self.method_label.pack(anchor="w", padx=(110, 0), pady=(4, 0))

    def _build_tabs(self, body):
        self.tabs = ttk.Notebook(body)
        self.tabs.pack(side="left", fill="both", expand=True)

        steps_tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(steps_tab, text="Як працює метод (покроково)")
        self.step_bar = ttk.Frame(steps_tab)
        self.step_bar.pack(fill="x")
        self.picture = ttk.Label(steps_tab, anchor="center")
        self.picture.pack(fill="both", expand=True, pady=8)
        self.explanation = ttk.Label(steps_tab, wraplength=PICTURE_SIZE[0], justify="left",
                                     font=("Helvetica", 15))
        self.explanation.pack(fill="x")
        nav = ttk.Frame(steps_tab)
        nav.pack(pady=(8, 0))
        ttk.Button(nav, text="◀ Назад", command=lambda: self.move_step(-1)).pack(side="left")
        self.step_label = ttk.Label(nav, width=14, anchor="center")
        self.step_label.pack(side="left")
        ttk.Button(nav, text="Далі ▶", command=lambda: self.move_step(1)).pack(side="left")

        compare_tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(compare_tab, text="Порівняння методів")
        ttk.Label(compare_tab, font=("Helvetica", 13), text=(
            "Результат кожного методу на цьому фото: червоне — знайдено, зелене — правильна "
            "розмітка. IoU — частка збігу з правильною розміткою (1 — повний збіг).")).grid(
                row=0, column=0, columnspan=2, sticky="w")
        self.compare_cells = []
        for index, _ in enumerate(COMPARED):
            cell = ttk.Frame(compare_tab, padding=6)
            cell.grid(row=1 + index // 2, column=index % 2)
            picture = ttk.Label(cell)
            picture.pack()
            caption = ttk.Label(cell, font=("Helvetica", 14))
            caption.pack()
            self.compare_cells.append((picture, caption))

    def _build_side(self, body):
        side = ttk.Frame(body, padding=(12, 0, 0, 0), width=340)
        side.pack(side="left", fill="y")
        ttk.Label(side, text="Наскільки добре знайдено", font=("Helvetica", 16, "bold")).pack(
            anchor="w")
        ttk.Label(side, text="Збіг з правильною розміткою (IoU):").pack(anchor="w", pady=(8, 0))
        self.iou_bar = ttk.Progressbar(side, maximum=100, length=300)
        self.iou_bar.pack(anchor="w")
        self.iou_label = ttk.Label(side, font=("Helvetica", 22, "bold"))
        self.iou_label.pack(anchor="w")
        ttk.Label(side, foreground="gray", wraplength=320, text=(
            "1 — знайдене повністю збігається з правильною розміткою, 0 — жодного спільного "
            "пікселя.")).pack(anchor="w")
        self.scores_label = ttk.Label(side, justify="left", wraplength=320,
                                      font=("Helvetica", 13))
        self.scores_label.pack(anchor="w", pady=(10, 0))

        ttk.Label(side, text="Кольори на фото", font=("Helvetica", 14, "bold")).pack(
            anchor="w", pady=(14, 0))
        for color, text in (("#ff3030", "червоне — знайшов метод"),
                            ("#20c020", "зелене — правильна розмітка"),
                            ("#e0b000", "жовте — зона пошуку")):
            tk.Label(side, text=f"■ {text}", fg=color, font=("Helvetica", 13)).pack(anchor="w")

        self.settings_button = ttk.Button(side, text="▸ Налаштування для експериментів",
                                          command=self.toggle_settings)
        self.settings_button.pack(fill="x", pady=(16, 0))
        self.settings = ttk.Frame(side)

    def _build_params(self, name: str):
        """Поля параметрів методу; зміни діють лише у вікні, не в config.json."""
        for child in self.settings.winfo_children():
            child.destroy()
        self.entries = {}
        ttk.Label(self.settings, foreground="gray", wraplength=320, text=(
            "Параметри з config.json. Змініть число й натисніть «Застосувати» — побачите, як "
            "зміниться результат. Файл config.json при цьому не змінюється.")).pack(anchor="w")
        # Лише секція самого методу (p1…p4): спільні секції (hough, roi) — у config.json.
        for key, value in self.config.get(name, {}).items():
            row = ttk.Frame(self.settings)
            row.pack(fill="x")
            ttk.Label(row, text=f"{name}.{key}", width=20, font=("Menlo", 11)).pack(side="left")
            entry = ttk.Entry(row, width=16)
            entry.insert(0, json.dumps(value))
            entry.pack(side="left")
            self.entries[(name, key)] = entry
        if name not in self.config:
            ttk.Label(self.settings, text="У p0 параметрів немає.").pack(anchor="w")
        ttk.Button(self.settings, text="Застосувати", command=self.refresh).pack(
            fill="x", pady=(8, 2))
        ttk.Button(self.settings, text="Повернути як у config.json",
                   command=self.on_method).pack(fill="x")

    # ------------------------------------------------------------ вибір
    def groups(self) -> list[str]:
        """Групи вибірки в порядку sample.csv."""
        return list(dict.fromkeys(sample["group"] for sample in self.samples))

    def group_samples(self) -> list[dict]:
        """Фото обраної групи."""
        group = self.groups()[self.group.current()]
        return [sample for sample in self.samples if sample["group"] == group]

    def on_group(self):
        """Нова група — показуємо її перше фото."""
        self.photo_index = 0
        self.load_photo()

    def move_photo(self, delta: int):
        """Попереднє (−1) чи наступне (+1) фото групи, по колу."""
        self.photo_index = (self.photo_index + delta) % len(self.group_samples())
        self.load_photo()

    def on_method(self):
        """Новий метод: опис, його параметри, показ з першого кроку."""
        name = self.method.get()
        info = REGISTRY[name]
        self.method_label["text"] = f"{METHODS[name]}  Автор: {info.owner}."
        self._build_params(name)
        self.step.set(0)
        if self.image is not None:
            self.refresh()

    def toggle_settings(self):
        """Показати чи сховати поля налаштувань."""
        if self.settings.winfo_ismapped():
            self.settings.pack_forget()
            self.settings_button["text"] = "▸ Налаштування для експериментів"
        else:
            self.settings.pack(fill="x", pady=(6, 0))
            self.settings_button["text"] = "▾ Сховати налаштування"

    def current_config(self) -> dict:
        """Копія конфігурації зі значеннями з полів налаштувань."""
        config = copy.deepcopy(self.config)
        for (section, key), entry in self.entries.items():
            config[section][key] = json.loads(entry.get())
        return config

    # ------------------------------------------------------------ обробка
    def load_photo(self):
        """Читає фото й правильну розмітку, потім обробляє."""
        samples = self.group_samples()
        sample = samples[self.photo_index]
        path = self.sample_dir / sample["image"]
        self.photo_label["text"] = f"фото {self.photo_index + 1} з {len(samples)}"
        self.file_label["text"] = path.name
        try:
            self.image = load_rgb(path)
        except OSError as error:
            messagebox.showerror("Фото не прочиталося", str(error))
            return
        self.gt_lanes = read_lanes(lines_path(path))
        self.refresh()

    def refresh(self):
        """Обробляє фото обраним методом і всіма чотирма для порівняння."""
        try:
            config = self.current_config()
            self.steps, scores = process(self.image, self.gt_lanes, self.method.get(), config)
            compared = [process(self.image, self.gt_lanes, name, config) for name in COMPARED]
        except (ValueError, LookupError, ArithmeticError, TypeError) as error:
            # Невдалий параметр не повинен закривати вікно.
            messagebox.showerror("Не вдалося обробити", f"{type(error).__name__}: {error}")
            return
        self._show_steps()
        self._show_scores(scores)
        self._show_comparison(compared)

    def move_step(self, delta: int):
        """Попередній (−1) чи наступний (+1) крок методу."""
        if self.steps:
            self.step.set(min(max(self.step.get() + delta, 0), len(self.steps) - 1))
            self.show_step()

    def _show_steps(self):
        for child in self.step_bar.winfo_children():
            child.destroy()
        for index, (title, _, _) in enumerate(self.steps):
            ttk.Radiobutton(self.step_bar, text=f"{index + 1}. {title}", value=index,
                            variable=self.step, command=self.show_step,
                            style="Toolbutton").pack(side="left", padx=2)
        self.step.set(min(self.step.get(), len(self.steps) - 1))
        self.show_step()

    def show_step(self):
        """Показує картинку й пояснення обраного кроку."""
        title, array, text = self.steps[self.step.get()]
        self.pictures["step"] = to_photo(array, PICTURE_SIZE)
        self.picture["image"] = self.pictures["step"]
        self.explanation["text"] = f"Крок {self.step.get() + 1}. {title}. {text}"
        self.step_label["text"] = f"крок {self.step.get() + 1} з {len(self.steps)}"

    def _show_scores(self, scores: dict):
        self.iou_bar["value"] = scores["iou"] * 100
        self.iou_label["text"] = number(scores["iou"])
        self.scores_label["text"] = (
            f"Знайдено ліній: {scores['found']}\n"
            f"Смуг насправді: {scores['expected']}\n"
            f"Час обробки: {number(scores['time_ms'], 0)} мс\n\n"
            f"Додатково: PSNR {number(scores['psnr'], 1)} дБ, SSIM {number(scores['ssim'], 3)}. "
            "Вони рахують і фон, тому в порожньої відповіді p0 навіть вищі — головна оцінка IoU.")

    def _show_comparison(self, compared: list):
        best = max(scores["iou"] for _, scores in compared)
        for name, (picture, caption), (steps, scores) in zip(COMPARED, self.compare_cells,
                                                             compared):
            self.pictures[name] = to_photo(steps[-1][1], SMALL_SIZE)
            picture["image"] = self.pictures[name]
            mark = "  ★ найкращий" if 0 < best == scores["iou"] else ""
            caption["text"] = (f"{name} «{REGISTRY[name].title}»: IoU {number(scores['iou'])}, "
                               f"ліній {scores['found']} із {scores['expected']}{mark}")


def main():
    """Відкриває вікно з параметрами з ``config.json``."""
    config = load_config()
    if not project_path(config["data"]["sample_dir"]).is_dir():
        raise SystemExit("Немає фото в data/sample: розпакуйте sample.zip (README)")
    root = tk.Tk()
    App(root, config)
    root.mainloop()


if __name__ == "__main__":
    main()
