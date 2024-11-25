# -*- coding: utf-8 -*-

"I. Compute and visualize ERDS maps"

"""Импорт библиотек"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import seaborn as sns
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test                    

"""Загрузка и предварительная обрабатка данных. 
Используются прогоны 4, 8 и 12 от субъекта 1 
(эти прогоны содержат образы движений правой и левой руки)."""

fnames = eegbci.load_data(subject=1, runs=(4, 8, 12))
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in fnames])

raw.rename_channels(lambda x: x.strip('.'))  #удалить точки из названий каналов

events, _ = mne.events_from_annotations(raw, event_id=dict(T1=1, T2=2))

"""Cоздание 5-секундных эпох вокруг нужных событий"""

tmin, tmax = -1, 4
event_ids = dict(right=1, left=2)

epochs = mne.Epochs(raw, events, event_ids, tmin - 0.5, tmax + 0.5,
                   picks=('C3', 'Cz', 'C4'), baseline=None, preload=True)

"""Установка подходящих значений для расчета карт ERDS"""

freqs = np.arange(2, 36)  # частоты в диапазоне от 2 до 35 Гц
vmin, vmax = -1.5, 1.5  # мин и макс значения на графике
baseline = (-1, 0)  # базовый интервал в сек
cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
              buffer_size=None, out_type='mask')

"""Декомпозиция время/частота по всем эпохам."""

tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs, use_fft=True,
                     return_itc=False, average=False, decim=2)
#Частотно-временное представление (TRF)
tfr.crop(tmin, tmax).apply_baseline(baseline, mode="percent")
#Обрезать данные до заданного временного интервала

for event in event_ids:
    # выбираем нужные эпохи для визуализации
    tfr_ev = tfr[event]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4),
                             gridspec_kw={"width_ratios": [10, 10, 10, 1]})
    for ch, ax in enumerate(axes[:-1]):  # перечисление для каждого канала
        # положит. кластеры
        _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch], tail=1, **kwargs)
        # отриц. кластеры
        _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch], tail=-1, **kwargs)

        # сохраняем кластеры с p <= 0,05 из объединенных кластеров 
        # двух независимых тестов
        c = np.stack(c1 + c2, axis=2)  # объединяем кластеры
        p = np.concatenate((p1, p2))  # комбинируем p-значения
        mask = c[..., p <= 0.05].any(axis=-1)

        # график TRF (ERDS карта с маскировкой)
        tfr_ev.average().plot([ch], cmap="RdBu", cnorm=cnorm, axes=ax,
                              colorbar=False, show=False, mask=mask,
                              mask_style="mask")

        ax.set_title(epochs.ch_names[ch], fontsize=10)
        ax.axvline(0, linewidth=1, color="black", linestyle=":")  # событие
        if ch != 0:
            ax.set_ylabel("")
            ax.set_yticklabels("")
    fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
    fig.suptitle(f"ERDS ({event})")
    plt.show()
    
"""Экспорт данных в файл DataFrame"""

df = tfr.to_data_frame(time_format=None)
df.head()

"""Построение доверительных интервалов"""

df = tfr.to_data_frame(time_format=None, long_format=True)

# Сопоставление с частотными диапазонами:
freq_bounds = {'_': 0,
               'delta': 3,
               'theta': 7,
               'alpha': 13,
               'beta': 35,
               'gamma': 140}
df['band'] = pd.cut(df['freq'], list(freq_bounds.values()),
                    labels=list(freq_bounds)[1:])

# Фильтруем, чтобы сохранить только соответствующие диапазоны частот:
freq_bands_of_interest = ['delta', 'theta', 'alpha', 'beta']
df = df[df.band.isin(freq_bands_of_interest)]
df['band'] = df['band'].cat.remove_unused_categories()

# Упорядочим каналы для построения графика:
df['channel'] = df['channel'].cat.reorder_categories(('C3', 'Cz', 'C4'),
                                                     ordered=True)

g = sns.FacetGrid(df, row='band', col='channel', margin_titles=True)
g.map(sns.lineplot, 'time', 'value', 'condition', n_boot=10)
axline_kw = dict(color='black', linestyle='dashed', linewidth=0.5, alpha=0.5)
g.map(plt.axhline, y=0, **axline_kw)
g.map(plt.axvline, x=0, **axline_kw)
g.set(ylim=(None, 1.5))
g.set_axis_labels("Time (s)", "ERDS (%)")
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.add_legend(ncol=2, loc='lower center')
g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)
