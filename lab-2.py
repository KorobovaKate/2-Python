import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from scipy.fft import dct
import time 

def main():

    start_time = time.time()

    #Чтение WAV-файла
    filename = r"C:\Users\user\Desktop\Python\2lab\15.wav"
    sample_rate, data = wavfile.read(filename)

    #Вывод информации о файле
    print(f"Имя файла: {filename}")
    print(f"Частота дискретизации: {sample_rate} Гц")
    print(f"Количество каналов: {1 if data.ndim == 1 else data.shape[1]}")
    print(f"Длительность: {len(data) / sample_rate:.2f} секунд")
    print(f"Количество отсчётов: {len(data)}")
    print(f"Тип данных: {data.dtype}")
    print(f"Диапазон амплитуд: [{data.min()}, {data.max()}]")

    #2.2 Визуализация сигнала
    plt.figure(figsize=(10, 4))
    time.axis = np.arange(len(data)) / sample_rate

    plt.plot(time.axis, data, linewidth=0.5)
    plt.title('Аудиосигнал: Коробова Екатерина')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('signal_time_domain.png', dpi=300)
    print("\nГрафик сохранён как 'signal_time_domain.png'")
    plt.show()


    #2.1 Визуализация дискретных отсчетов
    print("\n" + "=" * 50)
    print("ГРАФИК 1: Дискретные отсчеты (Ввод с клавиатуры)")
    print("=" * 50)
        
    try:
        user_input = input("Введите количество отсчетов для отображения (целое число): ")
        n_samples = int(user_input)
        
        if n_samples <= 0:
            print("Число должно быть положительным. Используем 1000 по умолчанию.")
            n_samples = 1000
            
    except ValueError:
        print("Ошибка: вы ввели не число. Будет отображено 1000 отсчетов.")
        n_samples = 1000

    if n_samples > len(data):
        print(f"В файле всего {len(data)} отсчетов. Отображаем все.")
        n_samples = len(data)

    data_short = data[:n_samples]
    time_short = np.arange(n_samples) / sample_rate

    plt.figure(figsize=(12, 5))
    plt.plot(time_short, data_short, linestyle='--', marker='*', markersize=4, 
                color='purple', linewidth=1)

    plt.title(f'График первых {n_samples} отсчетов', fontsize=14)
    plt.xlabel('Время (с)', fontsize=12)
    plt.ylabel('Амплитуда', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot_2_1_keyboard.png', dpi=300)
    plt.show()


    #2.3 Визуализация Спектрального анализа 
    print("\n" + "=" * 50)
    print("ДКП")
    print("=" * 50)

    # Применяем DCT типа 2
    dct_coefficients = dct(data, type=2, norm='ortho')

    print(f"Количество коэффициентов ДКП: {len(dct_coefficients)}")
    print(f"Диапазон коэффициентов: [{dct_coefficients.min():.2f}, {dct_coefficients.max():.2f}]")

    n_coefficients = min(1000, len(dct_coefficients))
    dct_to_plot = dct_coefficients[:n_coefficients]
    freq_indices = np.arange(n_coefficients)

    plt.figure(figsize=(18, 5))

    plt.plot(freq_indices, dct_to_plot, linestyle='--', marker='*', markersize=2, 
            linewidth=0.8, color='red', alpha=0.7)
    plt.title('Дискретное косинусное преобразование (ДКП) сигнала', fontsize=14)
    plt.xlabel('Номер коэффициента ДКП', fontsize=12)
    plt.ylabel('Амплитуда коэффициента', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dct_spectrum.png', dpi=300, bbox_inches='tight')
    plt.show()


    #2.4 Гисторграмма амплитуд 
    print("Гистограмма распределения амплитуд")
    print("=" * 50)

    plt.figure(figsize=(10, 5))

    plt.hist(data, bins=100, color='green', edgecolor='black', alpha=0.7)

    plt.title('Гистограмма амплитуд аудиосигнала', fontsize=14)
    plt.xlabel('Значение амплитуды', fontsize=12)
    plt.ylabel('Количество отсчетов (частота)', fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('histogram_amplitude.png', dpi=300, bbox_inches='tight')
    plt.show()

    end_time = time.time()
    execution_time = end_time - start_time

    print("\n" + "=" * 50)
    print(f"Время выполнения программы: {execution_time:.3f} секунд")
    print("=" * 50)


if __name__ == "__main__":
    main()