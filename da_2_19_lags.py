"""
da_2_19_lags.py
Создание лаговых признаков.
"""

import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_time_series(n: int, seed: int) -> pd.DataFrame:
    """
    Создаёт синтетический временной ряд: тренд + синус + шум.

    :param n: Количество точек.
    :param seed: Сид генератора случайных чисел (для воспроизводимости).
    :return: DataFrame со столбцами ['time', 'value'].
    """
    if n <= 0:
        raise ValueError("n должно быть положительным числом")

    np.random.seed(seed)
    time_idx = np.arange(n)
    values = 0.05 * time_idx + np.sin(time_idx / 3) + np.random.normal(0, 0.3, n)
    return pd.DataFrame({"time": time_idx, "value": values})


def add_lags(df: pd.DataFrame, lags: int) -> pd.DataFrame:
    """
    Добавляет лаговые признаки (t-1..t-lags) и удаляет строки с NaN.
    """
    if "value" not in df.columns:
        raise KeyError("В DataFrame должен быть столбец 'value'")
    if lags <= 0:
        raise ValueError("lags должен быть >= 1")

    out = df.copy()
    for i in range(1, lags + 1):
        out[f"lag{i}"] = out["value"].shift(i)
    out = out.dropna().reset_index(drop=True)
    return out


def plot_lag_scatter(df: pd.DataFrame, lag_col: str = "lag1") -> None:
    """
    Рисует диаграмму рассеяния: value(t) vs lag(t-k).

    :param df: DataFrame с колонками 'value' и lag_col.
    :param lag_col: Имя лаговой колонки (по умолчанию 'lag1').
    """
    if lag_col not in df.columns:
        raise KeyError(f"Колонка '{lag_col}' отсутствует в DataFrame")

    plt.figure(figsize=(6, 4))
    plt.scatter(df[lag_col], df["value"], alpha=0.75, edgecolor="k")
    plt.title(f"Зависимость value от {lag_col}")
    plt.xlabel(lag_col)
    plt.ylabel("value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    """
    Парсит аргументы командной строки.
    """
    p = argparse.ArgumentParser(
        description="DA-2-19: создание лаговых признаков для временного ряда."
    )
    p.add_argument("--n_points", type=int, default=100, metavar="N",
                   help="Количество точек во временном ряде (по умолчанию 100).")
    p.add_argument("--lags", type=int, default=2, metavar="K",
                   help="Колличество лагов (по умолчанию 2).")
    p.add_argument("--seed", type=int, default=None, metavar="SEED",
                   help="Сид генератора. По умолчанию берётся из текущего времени.")
    p.add_argument("--no-plot", action="store_true",
                   help="Не показывать график (только расчёты/вывод).")
    return p.parse_args()


def main() -> None:
    """
    Основная функция скрипта.

    1. Парсит аргументы командной строки.
    2. Генерирует синтетический временной ряд (с сидом из аргумента или текущего времени).
    3. Добавляет указанные лаговые признаки.
    4. Выводит первые строки DataFrame и вычисляет корреляцию между value и lag1.
    5. По умолчанию строит scatter-график зависимости value от lag1,
       если не передан флаг --no-plot.
    6. Все ошибки выполнения обрабатываются с выводом читаемого сообщения.
    """
    args = parse_args()

    # сид по умолчанию — из текущего времени (стабильно приводим к диапазону uint32)
    seed = args.seed if args.seed is not None else int(time.time()) & 0xFFFFFFFF

    try:
        df = generate_time_series(n=args.n_points, seed=seed)
        df_lagged = add_lags(df, lags=args.lags)

        # выводим первые строки и корреляцию с lag1 (если он есть)
        print(df_lagged.head())
        if "lag1" in df_lagged.columns:
            corr = df_lagged["value"].corr(df_lagged["lag1"])
            print(f"Корреляция между value и lag1: {corr:.3f}")

        # график можно отключить флагом --no-plot
        if not args.no_plot:
            plot_lag_scatter(df_lagged, lag_col="lag1")

    except Exception as e:
        print(f"[Ошибка] {e}")


if __name__ == "__main__":
    main()
