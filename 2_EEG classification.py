# -*- coding: utf-8 -*-

"II. Classification RE and IM"

"""Импорт библиотек"""

import os
import pyedflib
import mne
import uuid
import pandas as pd

import numpy as np
import torch

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Игнорируем ошибки
import warnings
warnings.filterwarnings("ignore")

"""Создание папки для хранения набор данных"""

ROOT_PATH = 'eeg-motor-movementimagery-dataset-1.0.0/files'
ROOT_PATH_DATA = os.path.join(ROOT_PATH, 'data')
ROOT_DATA = os.listdir(ROOT_PATH_DATA)
# Записи с реальными движениями
REC_NUM_RE = [3, 7, 11]
# Записи с воображаемыми движениями
REC_NUM_IM = [4, 8, 12]

# Создаем словарь с указанием меток аннотаций
lb_dec = {'T0': -1, 'T1': 0, 'T2': 1}

"""Предварительная загрузка одной записи"""

data_record = mne.io.read_raw_edf(os.path.join(ROOT_PATH_DATA, 'S001', 'S001R03.edf'), verbose=False)

# Каналы согласно расположению на шапке. В обучение пойдут симметричные пары C3 и C4
ch_names = ['Fc5.', 'Fc3.', 'Fc1.', 'Fc2.', 'Fc4.', 'Fc6.',
            'C5..', 'C3..', 'C1..', 'C2..', 'C4..', 'C6..',
            'Cp5.', 'Cp3.', 'Cp1.', 'Cp2.', 'Cp4.', 'Cp6.']

# Находим индексы этих каналов
ch_index = [data_record.ch_names.index(ch) for ch in ch_names]

"""Хранение промежуточной информации о данных"""

os.mkdir(os.path.join(ROOT_PATH, 'LR_data'))
os.mkdir(os.path.join(ROOT_PATH, 'LR_data', 'T1', ))
os.mkdir(os.path.join(ROOT_PATH, 'LR_data', 'T2', ))

files_data = pd.DataFrame(columns=['path', 'label'])

# Создание и сохранение в памяти метода для обработки
ica = mne.preprocessing.ICA(n_components=20, random_state=42)

"""Генерация набора данных"""

# Перебираем всех пациентов для генерации датасета. 
# Создание списков и предварительная обработка данных ЭЭГ каждого испытуемого
for patient in tqdm(ROOT_DATA):
    # У этих испытуемых неправильные временные метки
    if patient in ['S088', 'S092', 'S100']: continue

    os.mkdir(os.path.join(ROOT_PATH, 'LR_data', patient,))
    os.mkdir(os.path.join(ROOT_PATH, 'LR_data', patient, 'T1'))
    os.mkdir(os.path.join(ROOT_PATH, 'LR_data', patient, 'T2'))


    for record in os.listdir(os.path.join(ROOT_PATH_DATA, patient)):
        num_record = int(record[5:7])
        # здесь указываем задачу для классификации
        if num_record in REC_NUM_RE and 'event' not in record:

            data_record = mne.io.read_raw_edf(os.path.join(ROOT_PATH_DATA, patient, record), verbose=False, preload=True)

            # Применение полосового фильтра для удаления нежелательных частот
            data_record.filter(l_freq=8, h_freq=40, verbose=False)

            # Применение ICA для идентификации и удаления независимых компонентов,
            # представляющих шум или артефакты
            ica.fit(data_record, verbose=False)
            eeg_data_ica = ica.apply(data_record, verbose=False)

            data = eeg_data_ica.get_data()
            record_labels = eeg_data_ica.annotations
            
            # Предварительно обработанные данные сегментируем на несколько 
            # 4-секундных выборок, каждая из которых помечена соответствующим типом движения
            for label in record_labels:
                onset = label['onset']
                duration = label['duration']
                description = label['description']
                orig_time = label['orig_time']
                # print(int(onset * 160), int((onset + duration) * 160), data.shape)
                target_record = data[:, int(onset * 160) : int((onset * 160) + 640)]

                if description == 'T0': continue

                uuid_name = str(uuid.uuid4())

                os.mkdir(os.path.join(ROOT_PATH, 'LR_data', patient, description, uuid_name))
                for ch, idx in zip(ch_names, ch_index):
                    path_save = os.path.join(ROOT_PATH, 'LR_data', patient, description, uuid_name, ch + '_signal' +'.npy')

                    np.save(path_save, target_record[idx])

                    files_data = pd.concat((files_data, pd.DataFrame({'path': os.path.join('LR_data', patient, description, uuid_name),
                                                                      'label': description}, index=[0])), ignore_index=True)

# Запустить при первом запуске, затем закоментить
# files_data.to_csv(os.path.join(ROOT_PATH, 'files_data.csv'))

files_data = pd.read_csv(os.path.join(ROOT_PATH, 'files_data.csv'))
files_data

"""Подготовка к обучению модели"""

# Определение набора данных испытуемого для загрузки данных партиями во время обучения модели
class EEGDataset(Dataset):

    def __init__(self, df, channels, path):
        self.df = df
        self.channels = channels
        self.path = path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['label']
        full_paths = [os.path.join(self.path, path, ch + '_signal.npy') for ch in self.channels]
        data = [np.load(pth) for pth in full_paths]

        data = (data - np.min(data)) / (np.max(data) - np.min(data))

        return torch.Tensor(np.array([data])), lb_dec[label]

# Разделение набора данных на два подмножества: train и test
# 20% данных будут использоваться для тестирования
train, test = train_test_split(files_data, test_size=0.2, shuffle=True, random_state=42)

train_loader = DataLoader(EEGDataset(train, ['C3..', 'C4..'], ROOT_PATH), batch_size=4, shuffle=True)
test_loader = DataLoader(EEGDataset(test, ['C3..', 'C4..'], ROOT_PATH), batch_size=4, shuffle=True)

next(iter(train_loader))[1]

"""Обучение и классификация"""

# Наложение сверточных слоев
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 25, (1, 11))

        self.conv2 = nn.Conv2d(25, 25, (2, 1))
        self.pool2 = nn.MaxPool2d((1, 3))

        self.conv3 = nn.Conv2d(25, 50, (1, 11))
        self.pool3 = nn.MaxPool2d((1, 3))

        self.conv4 = nn.Conv2d(50, 100, (1, 11))
        self.pool4 = nn.MaxPool2d((1, 3))

        self.conv5 = nn.Conv2d(100, 200, (1, 11))
        self.pool5 = nn.MaxPool2d((1, 2))

        self.fc1 = nn.Linear(800, 100)
        self.fc2 = nn.Linear(100, 2)

        self.dropout = nn.Dropout2d(p=0.5)
        self.batchNorm2 = nn.BatchNorm2d(25)
        self.batchNorm4 = nn.BatchNorm2d(100)

    def forward(self, x):
        x = self.dropout(F.leaky_relu(self.conv1(x)))
        x = self.batchNorm2(self.pool2(F.leaky_relu(self.conv2(x))))
        x = self.dropout(self.pool3(F.leaky_relu(self.conv3(x))))
        x = self.batchNorm4(self.pool4(F.leaky_relu(self.conv4(x))))
        x = self.dropout(self.pool5(F.leaky_relu(self.conv5(x))))

        # x = x.view(x.shape[0], -1)
        x = torch.flatten(x, 1)

        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)

        return x

net = Net()

# Инициализация функции потерь и оптимизатора для обучения нейронной сети
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=8e-5,)

# Запуск цикла обучения модели для определенного количества эпох
# В нашем случае 20 эпох
for epoch in range(20):

    running_loss = 0.0
    for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
        # Получить входные данные; данные - это список [inputs, labels]
        inputs, labels = data
        
        # Обнуление градиентов параметров
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Вывод статистики
        running_loss += loss.item()
        if i % 500 == 500 - 1:    # печать каждых 200 мини-пакетов
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.5f}')
            running_loss = 0.0

    true_labels = []
    pred_labels = []
    # Валидация
    for i, data in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
        # Получить входные данные; данные - это список [inputs, labels]
        inputs, labels = data
        outputs = net(inputs)

        pred_labels.extend(np.argmax(outputs.detach().numpy(), 1).tolist())
        true_labels.extend(labels.tolist())

    print(accuracy_score(true_labels, pred_labels))

print('Finished Training')
