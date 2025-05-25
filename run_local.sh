#!/bin/bash

# Директория для логов локальных запусков
LOG_DIR="logs_local"
rm -rf $LOG_DIR
mkdir -p $LOG_DIR

REPEATS=10
# Используем те же значения для количества процессов/потоков, что и в оригинальном скрипте
# Вы можете изменить этот массив при необходимости
PROCESSES_OR_THREADS=(1 2 4 8 16) 

MPI_EXECUTABLE="combined_mpi_taylor"
OMP_EXECUTABLE="combined_omp_taylor"

MPI_SOURCE="${MPI_EXECUTABLE}.c"
OMP_SOURCE="${OMP_EXECUTABLE}.c"

# Переменные окружения для OpenMP (можно оставить, они влияют на производительность)
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_SCHEDULE=dynamic,64 # Пример, можно настроить или убрать

# Компиляция
echo "Компиляция MPI версии ($MPI_SOURCE)..."
mpicc -o $MPI_EXECUTABLE $MPI_SOURCE -lm -O3
if [ $? -ne 0 ]; then
    echo "Ошибка компиляции MPI версии. Выход."
    exit 1
fi

echo "Компиляция OpenMP версии ($OMP_SOURCE)..."
gcc -fopenmp -o $OMP_EXECUTABLE $OMP_SOURCE -lm -O3
if [ $? -ne 0 ]; then
    echo "Ошибка компиляции OpenMP версии. Выход."
    exit 1
fi

# Функция для запуска MPI задач
run_mpi_jobs_local() {
    echo "-------------------------------------"
    echo "Запуск MPI задач локально..."
    echo "-------------------------------------"
    for procs in "${PROCESSES_OR_THREADS[@]}"; do
        echo "Запуск MPI с $procs процессами..."
        for ((i = 1; i <= REPEATS; i++)); do
            LOG_FILE="${LOG_DIR}/output_MPI_${procs}_procs_run${i}_123.out"
            echo "  Попытка ${i}/${REPEATS}, лог: ${LOG_FILE}"
            # Запуск mpirun с указанием количества процессов
            # --bind-to core и --map-by core полезны для привязки процессов к ядрам
            mpirun -np "$procs" --bind-to core --map-by core "./$MPI_EXECUTABLE" > "$LOG_FILE" 2>&1
            if [ $? -ne 0 ]; then
                echo "    Ошибка выполнения MPI для $procs процессов, попытка $i. См. $LOG_FILE"
            fi
            sleep 0.1 # Небольшая пауза между запусками
        done
    done
}

# Функция для запуска OpenMP задач
run_omp_jobs_local() {
    echo "-------------------------------------"
    echo "Запуск OpenMP задач локально..."
    echo "-------------------------------------"
    for threads in "${PROCESSES_OR_THREADS[@]}"; do
        echo "Запуск OpenMP с $threads потоками..."
        export OMP_NUM_THREADS=$threads # Установка количества потоков для OpenMP
        for ((i = 1; i <= REPEATS; i++)); do
            LOG_FILE="${LOG_DIR}/output_OMP_${threads}_threads_run${i}_123.out"
            echo "  Попытка ${i}/${REPEATS}, OMP_NUM_THREADS=${threads}, лог: ${LOG_FILE}"
            # Запуск OpenMP исполняемого файла
            "./$OMP_EXECUTABLE" > "$LOG_FILE" 2>&1
            if [ $? -ne 0 ]; then
                echo "    Ошибка выполнения OpenMP для $threads потоков, попытка $i. См. $LOG_FILE"
            fi
            sleep 0.1 # Небольшая пауза между запусками
        done
    done
}

# Запуск задач
run_mpi_jobs_local
run_omp_jobs_local

echo "-------------------------------------"
echo "Все локальные задачи завершены."
echo "Логи сохранены в директории: $LOG_DIR"
echo "-------------------------------------"