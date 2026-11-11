# Дані: CULane і наша вибірка

## Звідки

[CULane](https://xingangpan.github.io/projects/CULane.html) — 133 235 кадрів з
камер шести автомобілів у Пекіні, розмір 1640 × 590. Розмічено до 4 смуг на
кадр, зокрема смуги, закриті машинами. Тестова частина поділена на звичайну
групу й 8 складних.

Ліцензія: лише некомерційне використання, обов'язкове цитування, **поширювати
набір і його змінені копії заборонено**. Тому фото не комітимо.

Цитування для звіту:

> Pan X., Zhan X., Shi J., Luo P., Wang X., Tang X. Spatial As Deep: Spatial CNN for
> Traffic Scene Understanding // AAAI Conference on Artificial Intelligence. —
> 2018.

## Що завантажити (задача T0.3)

Зі сторінки CULane (Google Drive або Baidu):

| Архів | Навіщо |
|---|---|
| `list.tar.gz` | списки тестових груп `list/test_split/test*.txt` |
| `driver_37_30frame.tar.gz` | тестові кадри |
| `driver_100_30frame.tar.gz` | тестові кадри |
| `driver_193_90frame.tar.gz` | тестові кадри |

Архіви великі. Розпакуй у `data/CULane/`:

```
data/CULane/
  list/test_split/test0_normal.txt …
  driver_37_30frame/<кліп>.MP4/<кадр>.jpg
  driver_37_30frame/<кліп>.MP4/<кадр>.lines.txt
  …
```

Після розпакування перевір, що поруч із `.jpg` лежить `.lines.txt`. Якщо
файлів розмітки немає, напиши в чат команди — `make_sample.py` без них не
працює.

## Формат розмітки

`<кадр>.lines.txt` — рядок на смугу: `x1 y1 x2 y2 x3 y3 …` (дійсні числа,
пікселі кадру). Порожній файл — на кадрі немає розмічених смуг.

## Наша вибірка (задача F3)

```bash
python -m tools.make_sample
```

Скрипт для кожної групи з `config.json → data.groups` бере `per_group` (20)
випадкових кадрів з `seed`, у яких є хоча б одна смуга, і копіює їх разом із
`.lines.txt` у `data/sample/<група>/`. Назва файлу: шлях у CULane, де `/`
замінено на `__`. Список пише в `data/sample.csv` (`group,image,source`) —
він у git.

| Група | Список CULane | Фото |
|---|---|---|
| normal — звичайна дорога | `test0_normal.txt` | 20 |
| crowd — щільний трафік | `test1_crowd.txt` | 20 |
| shadow — тіні | `test3_shadow.txt` | 20 |
| curve — поворот | `test6_curve.txt` | 20 |
| night — ніч | `test8_night.txt` | 20 |

## Передати вибірку Максиму

Фояк пакує `data/sample/` у `sample.zip` і передає приватно (Google Drive з
доступом лише для команди). Максим розпаковує в `data/sample/`. Публічно
архів не викладаємо — ліцензія.
