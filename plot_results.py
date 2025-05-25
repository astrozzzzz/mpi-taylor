import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

RESULTS_DIR = r"c:\Nikita\!ВУЗ\ПВС\mpi-taylor\logs_local"
OUTPUT_PNG_FILE = "execution_time_comparison.png"

def parse_time_from_line(line):
    """Извлекает время из строки вида 'Затраченное время: X.XXXXXX секунд'"""
    # Обновленное регулярное выражение для соответствия вашему формату вывода
    match = re.search(r"Затраченное время:\s*(\d+(\.\d+)?)\s*секунд", line)
    if match:
        return float(match.group(1))
    return None

def main():
    mpi_times = defaultdict(list)
    omp_times = defaultdict(list)

    if not os.path.isdir(RESULTS_DIR):
        print(f"Директория с результатами не найдена: {RESULTS_DIR}")
        return

    for filename in os.listdir(RESULTS_DIR):
        if filename.startswith("output_") and filename.endswith(".out"):
            filepath = os.path.join(RESULTS_DIR, filename)
            
            num_cores = 0
            run_type = ""

            mpi_match = re.match(r"output_MPI_(\d+)_procs_run\d+_\d+\.out", filename)
            if mpi_match:
                num_cores = int(mpi_match.group(1))
                run_type = "MPI"
            else:
                omp_match = re.match(r"output_OMP_(\d+)_threads_run\d+_\d+\.out", filename)
                if omp_match:
                    num_cores = int(omp_match.group(1))
                    run_type = "OMP"
            
            if not run_type or num_cores == 0:
                # Пропускаем файлы, которые не соответствуют ожидаемому формату именования
                # или если не удалось извлечь количество ядер/потоков.
                # print(f"Пропущен файл (не соответствует формату): {filename}")
                continue

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    file_content = f.read() # Читаем весь файл, т.к. нужная строка может быть где угодно
                    time_val = parse_time_from_line(file_content)
                    if time_val is not None:
                        if run_type == "MPI":
                            mpi_times[num_cores].append(time_val)
                        elif run_type == "OMP":
                            omp_times[num_cores].append(time_val)
                    # else:
                        # print(f"Время не найдено в файле: {filename}")

            except Exception as e:
                print(f"Ошибка при чтении файла {filepath}: {e}")

    if not mpi_times and not omp_times:
        print("Не найдено данных для построения графика.")
        return

    avg_mpi_times = {k: np.mean(v) for k, v in mpi_times.items() if v}
    avg_omp_times = {k: np.mean(v) for k, v in omp_times.items() if v}

    mpi_procs_sorted = sorted(avg_mpi_times.keys())
    mpi_avg_times_sorted = [avg_mpi_times[p] for p in mpi_procs_sorted]

    omp_threads_sorted = sorted(avg_omp_times.keys())
    omp_avg_times_sorted = [avg_omp_times[t] for t in omp_threads_sorted]

    plt.figure(figsize=(12, 7))

    if mpi_avg_times_sorted:
        plt.plot(mpi_procs_sorted, mpi_avg_times_sorted, marker='o', linestyle='-', label='MPI (combined_mpi_taylor.c)')
        print(f"MPI данные: Процессы={mpi_procs_sorted}, Среднее время={mpi_avg_times_sorted}")

    if omp_avg_times_sorted:
        plt.plot(omp_threads_sorted, omp_avg_times_sorted, marker='s', linestyle='--', label='OpenMP (combined_omp_taylor.c)')
        print(f"OpenMP данные: Потоки={omp_threads_sorted}, Среднее время={omp_avg_times_sorted}")

    plt.title('Сравнение времени выполнения MPI vs OpenMP (Taylor Series)')
    plt.xlabel('Количество процессов / потоков')
    plt.ylabel('Среднее время выполнения (секунды)')
    
    # Убедимся, что все значения x на оси и они целые
    all_x_values = sorted(list(set(mpi_procs_sorted + omp_threads_sorted)))
    if all_x_values:
        plt.xticks(all_x_values)
        
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    
    try:
        plt.savefig(OUTPUT_PNG_FILE)
        print(f"График сохранен в {os.path.abspath(OUTPUT_PNG_FILE)}")
    except Exception as e:
        print(f"Ошибка при сохранении графика: {e}")
    
    # plt.show()

if __name__ == "__main__":
    main()
