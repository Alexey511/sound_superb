# SUPERB audio Classification with CNN

Проект по классификации звуков с использованием сверточных нейронных сетей (CNN).

## Цель
Классифицировать аудиофайлы по категориям (с классом "unknown") с помощью CNN.

## Результаты
- Recall: ~0.98 на целевых классах
- Метрики: Recall, Precision, F1 по классам, Confusion Matrix

## Структура
- 4_sound_eda.ipynb: Анализ данных (баланс классов, обработка длительности, создание спектрограмм)
- 4_sound_multiprocessing_NN.py: Создание даталоадера, обучение CNN. Чистый питон из-за использования многопоточности.
- 4_sound_results.ipynb: Визуализация метрик (accuracy, recall, precision, F1, confusion matrix)
- Папки с датасетом, спектрограммами и результатами не выгружены (весят ~100гб).

## Применяемые библиотеки
- numpy
- torch
- librosa
- matplotlib
- seaborn
- sklearn
- datasets
- tqdm
- Базовые библиотеки python

## Датасет
- Использован датасет SUPERB (Speech processing Universal PERformance Benchmark)
- Сайт: https://superbbenchmark.org
- Можно скачать с помощью 'from datasets import load_dataset' и 'dataset = load_dataset("superb", "ks")'
