#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h> // Замена mpi.h на omp.h

#define SIZE 3
#define N_TERMS 50000000
// #define NUM_MY_THREADS 1 // Определяем желаемое количество потоков - УДАЛЕНО

typedef struct {
    double data[SIZE][SIZE];
} Matrix;

void matrix_print(const char* label, const double matrix[SIZE][SIZE]);
Matrix matrix_add(const double matrix1[SIZE][SIZE], const double matrix2[SIZE][SIZE]);
Matrix matrix_multiply(const double matrix1[SIZE][SIZE], const double matrix2[SIZE][SIZE]);
void matrix_identity(double matrix[SIZE][SIZE]);
Matrix matrix_scalar_multiply(const double matrix[SIZE][SIZE], double scalar);
Matrix matrix_power(const double base_matrix[SIZE][SIZE], long exp);

// Функция matrix_sum_mpi_op удалена, так как она специфична для MPI

void matrix_print(const char* label, const double matrix[SIZE][SIZE]) {
    if (label != NULL && label[0] != '\0') {
        printf("%s:\n", label);
    }
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%10.6f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

Matrix matrix_add(const double matrix1[SIZE][SIZE], const double matrix2[SIZE][SIZE]) {
    Matrix result;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            result.data[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
    return result;
}

Matrix matrix_multiply(const double matrix1[SIZE][SIZE], const double matrix2[SIZE][SIZE]) {
    Matrix result;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            result.data[i][j] = 0.0;
            for (int k = 0; k < SIZE; k++) {
                result.data[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    return result;
}

void matrix_identity(double matrix[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

Matrix matrix_scalar_multiply(const double matrix[SIZE][SIZE], const double scalar) {
    Matrix result;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            result.data[i][j] = matrix[i][j] * scalar;
        }
    }
    return result;
}

Matrix matrix_power(const double base_matrix[SIZE][SIZE], long exp) {
    Matrix result_matrix_struct;
    matrix_identity(result_matrix_struct.data);

    Matrix current_power_struct;
    for(int r=0; r<SIZE; ++r) {
        for(int c=0; c<SIZE; ++c) {
            current_power_struct.data[r][c] = base_matrix[r][c];
        }
    }

    if (exp == 0) {
        return result_matrix_struct;
    }
    if (exp < 0) {
        fprintf(stderr, "Ошибка: Отрицательная степень в matrix_power не поддерживается.\n");
        exit(1); // Замена MPI_Abort на exit(1)
    }

    long current_exp = exp;
    while (current_exp > 0) {
        if (current_exp % 2 == 1) {
            result_matrix_struct = matrix_multiply(result_matrix_struct.data, current_power_struct.data);
        }
        if (current_exp / 2 > 0) {
             current_power_struct = matrix_multiply(current_power_struct.data, current_power_struct.data);
        }
        current_exp /= 2;
    }
    return result_matrix_struct;
}

int main(int argc, char *argv[]) {
    // Удалены переменные rank и num_procs из MPI
    // Удалены вызовы MPI_Init, MPI_Comm_rank, MPI_Comm_size

    double A[SIZE][SIZE] = {
        {0.2, 0.3, 0.5},
        {0.6, 0.1, 0.3},
        {0.2, 0.3, 0.1}
    };

    Matrix global_taylor_sum_struct;
    // Инициализация глобальной суммы нулями
    for(int i=0; i<SIZE; ++i) {
        for(int j=0; j<SIZE; ++j) {
            global_taylor_sum_struct.data[i][j] = 0.0;
        }
    }

    double start_time = omp_get_wtime(); // Используем таймер OpenMP

    // Устанавливаем количество потоков перед параллельной областью - УДАЛЕНО
    // omp_set_num_threads(NUM_MY_THREADS); 

    long M_jump_size = (long)sqrt((double)N_TERMS);
    if (M_jump_size == 0) M_jump_size = 1;

    Matrix A_pow_M_struct;
    // Вычисление A_pow_M_struct выполняется один раз, до параллельной области
    // Удалена проверка if (rank == 0)
    if (M_jump_size == 1) {
        for(int r=0; r<SIZE; ++r) for(int c=0; c<SIZE; ++c) A_pow_M_struct.data[r][c] = A[r][c];
    } else {
        A_pow_M_struct = matrix_power(A, M_jump_size);
    }
    // MPI_Bcast удален, так как A_pow_M_struct будет доступна всем потокам в общей памяти

    int final_num_threads_for_print = 0;

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads(); // Здесь num_threads будет равно NUM_MY_THREADS (или меньше, если система не может предоставить столько)

        #pragma omp master
        {
            final_num_threads_for_print = num_threads;
        }

        Matrix local_taylor_sum_struct; // Локальная сумма для каждого потока
        for(int i=0; i<SIZE; ++i) for(int j=0; j<SIZE; ++j) local_taylor_sum_struct.data[i][j] = 0.0;
        
        Matrix current_term_struct;    // Текущий член ряда для каждого потока
        double temp_matrix_thread[SIZE][SIZE]; // Временная матрица для каждого потока

        long k_start_idx, k_end_idx;
        long num_my_terms;

        // Распределение итераций (членов ряда) между потоками
        long base_count = N_TERMS / num_threads;
        long extra_count = N_TERMS % num_threads;

        if (thread_id < extra_count) {
            k_start_idx = thread_id * (base_count + 1) + 1;
            num_my_terms = base_count + 1;
        } else {
            k_start_idx = thread_id * base_count + extra_count + 1;
            num_my_terms = base_count;
        }
        k_end_idx = k_start_idx + num_my_terms - 1;

        if (num_my_terms == 0) {
            k_start_idx = 0;
            k_end_idx = -1; // Поток не будет выполнять итерации
        }
        
        // Предварительное вычисление начального члена T_{k_start_idx - 1} для каждого потока
        Matrix current_T_val_struct_thread; 
        double my_initial_term_T_km1_thread[SIZE][SIZE];
        long target_k_for_precalc = k_start_idx - 1;

        if (target_k_for_precalc <= 0) {
            matrix_identity(my_initial_term_T_km1_thread);
        } else {
            matrix_identity(current_T_val_struct_thread.data);
            long current_k_val = 0;

            // A_pow_M_struct и A доступны всем потокам (общая память, только чтение)
            for (long j = 0; j < target_k_for_precalc / M_jump_size; ++j) {
                double scalar_prod_inv = 1.0;
                for (long l = 1; l <= M_jump_size; ++l) {
                    scalar_prod_inv /= (double)(current_k_val + l);
                }
                Matrix temp_prod_struct = matrix_multiply(current_T_val_struct_thread.data, A_pow_M_struct.data);
                current_T_val_struct_thread = matrix_scalar_multiply(temp_prod_struct.data, scalar_prod_inv);
                current_k_val += M_jump_size;
            }

            for (long l = 0; l < target_k_for_precalc % M_jump_size; ++l) {
                current_k_val += 1;
                Matrix temp_prod_struct = matrix_multiply(current_T_val_struct_thread.data, A);
                current_T_val_struct_thread = matrix_scalar_multiply(temp_prod_struct.data, 1.0 / (double)current_k_val);
            }
            for(int r=0; r<SIZE; ++r) for(int c=0; c<SIZE; ++c) my_initial_term_T_km1_thread[r][c] = current_T_val_struct_thread.data[r][c];
        }

        // Инициализация current_term_struct для основного цикла этого потока
        for(int r=0; r<SIZE; ++r) for(int c=0; c<SIZE; ++c) current_term_struct.data[r][c] = my_initial_term_T_km1_thread[r][c];

        // Основной цикл вычисления членов ряда для данного потока
        if (num_my_terms > 0) {
            for (long k = k_start_idx; k <= k_end_idx; ++k) {
                Matrix term_A_prod_struct = matrix_multiply(current_term_struct.data, A);
                for (int r = 0; r < SIZE; ++r) for (int c = 0; c < SIZE; ++c) temp_matrix_thread[r][c] = term_A_prod_struct.data[r][c];
                
                Matrix actual_term_k_struct = matrix_scalar_multiply(temp_matrix_thread, 1.0 / (double)k);
                
                Matrix new_sum_struct = matrix_add(local_taylor_sum_struct.data, actual_term_k_struct.data);
                for (int r = 0; r < SIZE; ++r) for (int c = 0; c < SIZE; ++c) local_taylor_sum_struct.data[r][c] = new_sum_struct.data[r][c];
                
                for (int r = 0; r < SIZE; ++r) for (int c = 0; c < SIZE; ++c) current_term_struct.data[r][c] = actual_term_k_struct.data[r][c];
            }
        }

        // Добавление локальной суммы потока к глобальной сумме с использованием критической секции
        #pragma omp critical
        {
            Matrix temp_global_sum_struct = matrix_add(global_taylor_sum_struct.data, local_taylor_sum_struct.data);
            for (int r_crit = 0; r_crit < SIZE; ++r_crit) {
                for (int c_crit = 0; c_crit < SIZE; ++c_crit) {
                    global_taylor_sum_struct.data[r_crit][c_crit] = temp_global_sum_struct.data[r_crit][c_crit];
                }
            }
        }
    } // Конец #pragma omp parallel

    // MPI_Reduce и MPI_Op_free удалены

    // Добавление нулевого члена (единичной матрицы) к глобальной сумме
    // global_taylor_sum_struct теперь содержит сумму членов с T_1 до T_N_TERMS
    double identity_m[SIZE][SIZE];
    matrix_identity(identity_m);
    Matrix final_sum_struct = matrix_add(global_taylor_sum_struct.data, identity_m);
    for (int r = 0; r < SIZE; ++r) for (int c = 0; c < SIZE; ++c) global_taylor_sum_struct.data[r][c] = final_sum_struct.data[r][c];
    
    double end_time = omp_get_wtime(); // Используем таймер OpenMP

    // Вывод результатов (выполняется один раз)
    printf("============================================================\n");
    printf("Результаты вычисления экспоненты матрицы e^A (OpenMP):\n");
    printf("============================================================\n");
    matrix_print("Исходная матрица A", A);
    matrix_print("Результат e^A (аппроксимация рядом Тейлора)", global_taylor_sum_struct.data);
    printf("------------------------------------------------------------\n");
    printf("Затраченное время: %f секунд\n", end_time - start_time);
    printf("Количество членов ряда N_TERMS: %ld\n", (long)N_TERMS);
    printf("Размер матрицы SIZE: %d\n", SIZE);
    printf("Количество OpenMP потоков: %d\n", final_num_threads_for_print);
    printf("============================================================\n");

    // MPI_Finalize() удален
    return 0;
}
