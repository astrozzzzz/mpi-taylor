#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h> // Для инициализации случайными числами (если нужно)

// Размерность матрицы (например, 3x3)
#define DIM 3
// Количество членов в ряде Тейлора
#define N_TERMS 50000000

// Прототипы вспомогательных функций
double* allocate_matrix_1d(int rows, int cols);
void free_matrix_1d(double* matrix);
void initialize_matrix_zeros_1d(double* matrix, int rows, int cols);
void initialize_identity_matrix_1d(double* matrix, int dim);
void initialize_matrix_A_1d(double* matrix, int dim); // Пример инициализации матрицы A
void print_matrix_1d(const double* matrix, int rows, int cols, const char* title);
void matrix_multiply_1d(const double* A, const double* B, double* C, int dim); // C = A * B
void matrix_add_1d(double* A_accumulator, const double* B_term, int dim); // A_accumulator += B_term
void matrix_scalar_divide_1d(const double* A, double scalar, double* C, int dim); // C = A / scalar
void copy_matrix_1d(const double* src, double* dest, int dim);


int main(int argc, char* argv[]) {
    int world_rank, world_size;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (DIM <= 0 || N_TERMS <= 0) {
        if (world_rank == 0) {
            fprintf(stderr, "DIM (размерность) и N_TERMS (количество членов) должны быть положительными.\n");
        }
        MPI_Finalize();
        return 1;
    }

    double* matrix_A = allocate_matrix_1d(DIM, DIM);
    double* local_sum_A = allocate_matrix_1d(DIM, DIM); // Локальная сумма членов ряда для каждого процесса
    double* current_term = allocate_matrix_1d(DIM, DIM); // Текущий член ряда T_k (или T_{k-1} в зависимости от шага)
    double* temp_matrix_mult_result = allocate_matrix_1d(DIM, DIM); // Временная матрица для результата умножения

    if (world_rank == 0) {
        initialize_matrix_A_1d(matrix_A, DIM); // Инициализация матрицы A на процессе 0
        printf("Матрица A (DIM=%d):\n", DIM);
        print_matrix_1d(matrix_A, DIM, DIM, "Матрица A");
        printf("Количество членов ряда Тейлора: %d\n", N_TERMS);
        printf("Количество MPI процессов: %d\n", world_size);
    }

    start_time = MPI_Wtime(); // Засекаем время начала параллельных вычислений

    // Рассылка матрицы A всем процессам из процесса 0
    MPI_Bcast(matrix_A, DIM * DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Инициализация локальной суммы нулями
    initialize_matrix_zeros_1d(local_sum_A, DIM, DIM);

    // --- Определение блока членов ряда для текущего процесса ---
    // N_TERMS - это общее количество членов, включая T_0 (k=0 до N_TERMS-1)
    int num_terms_total = N_TERMS;
    int terms_per_rank_base = num_terms_total / world_size;
    int remainder_terms = num_terms_total % world_size;

    int my_start_term_index; // Начальный индекс k для этого процесса (включительно)
    int my_end_term_index;   // Конечный индекс k для этого процесса (не включительно)

    if (world_rank < remainder_terms) {
        // Первые 'remainder_terms' процессов получают на один член больше
        my_start_term_index = world_rank * (terms_per_rank_base + 1);
        my_end_term_index = my_start_term_index + (terms_per_rank_base + 1);
    } else {
        // Остальные процессы получают 'terms_per_rank_base' членов
        my_start_term_index = world_rank * terms_per_rank_base + remainder_terms;
        my_end_term_index = my_start_term_index + terms_per_rank_base;
    }

    // Для отладки можно раскомментировать:
    // printf("Rank %d: отвечает за члены ряда с индексами k от %d до %d (не включая)\n", world_rank, my_start_term_index, my_end_term_index);

    // --- Инициализация current_term для начала вычислений этого процесса ---
    // current_term будет хранить T_{k-1} перед вычислением T_k в цикле.
    // Начинаем с T_0 = I.
    initialize_identity_matrix_1d(current_term, DIM);

    // Если первый член, за который отвечает процесс (my_start_term_index), не T_0,
    // то нужно вычислить T_{my_start_term_index - 1}.
    // current_term сейчас T_0.
    if (my_start_term_index > 0) {
        for (int k_pre = 1; k_pre < my_start_term_index; ++k_pre) {
            // current_term на входе T_{k_pre - 1}
            matrix_multiply_1d(current_term, matrix_A, temp_matrix_mult_result, DIM);
            // temp_matrix_mult_result = T_{k_pre - 1} * A
            matrix_scalar_divide_1d(temp_matrix_mult_result, (double)k_pre, current_term, DIM);
            // current_term теперь T_{k_pre}
        }
        // После цикла, current_term = T_{my_start_term_index - 1}
    }
    // Если my_start_term_index == 0, current_term остался T_0.

    // --- Итеративное вычисление и суммирование членов ряда A^k/k! для ДАННОГО процесса ---
    // Цикл по k от my_start_term_index до my_end_term_index - 1.
    for (int k = my_start_term_index; k < my_end_term_index; ++k) {
        // На входе в итерацию:
        // - Если k = 0 (и my_start_term_index = 0), current_term = T_0. Это и есть нужный член.
        // - Если k > 0, current_term = T_{k-1}. Нужно вычислить T_k.

        if (k == 0) {
            // current_term это T_0. Добавляем его в локальную сумму.
            // Он останется T_0 для следующей итерации (k=1), где он будет T_{k-1}.
            matrix_add_1d(local_sum_A, current_term, DIM);
        } else {
            // current_term это T_{k-1}. Вычисляем T_k = (T_{k-1} * A) / k.
            matrix_multiply_1d(current_term, matrix_A, temp_matrix_mult_result, DIM);
            // temp_matrix_mult_result = T_{k-1} * A
            // Обновляем current_term до T_k (результат T_k записывается в current_term).
            matrix_scalar_divide_1d(temp_matrix_mult_result, (double)k, current_term, DIM);
            // current_term теперь T_k. Добавляем его в локальную сумму.
            matrix_add_1d(local_sum_A, current_term, DIM);
        }
        // После этой итерации current_term хранит T_k, который будет T_{ (k+1) - 1 } для следующей итерации.
    }

    double* final_exp_A = NULL; // Матрица для финального результата на процессе 0
    if (world_rank == 0) {
        final_exp_A = allocate_matrix_1d(DIM, DIM);
        // Нет необходимости инициализировать нулями, MPI_Reduce перезапишет значения
    }

    // Сбор и суммирование всех local_sum_A в final_exp_A на процессе 0
    MPI_Reduce(local_sum_A, final_exp_A, DIM * DIM, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    end_time = MPI_Wtime(); // Засекаем время окончания

    if (world_rank == 0) {
        printf("\nВычисленная матрица e^A:\n");
        print_matrix_1d(final_exp_A, DIM, DIM, "e^A");
        printf("\nВремя вычислений: %f секунд\n", end_time - start_time);
        free_matrix_1d(final_exp_A);
    }

    // Освобождение памяти
    free_matrix_1d(matrix_A);
    free_matrix_1d(local_sum_A);
    free_matrix_1d(current_term);
    free_matrix_1d(temp_matrix_mult_result);

    MPI_Finalize();
    return 0;
}

// --- Реализация вспомогательных функций ---

double* allocate_matrix_1d(int rows, int cols) {
    double* matrix = (double*)malloc(rows * cols * sizeof(double));
    if (!matrix) {
        perror("Ошибка выделения памяти для матрицы");
        MPI_Abort(MPI_COMM_WORLD, 1); // Аварийное завершение MPI программы
    }
    return matrix;
}

void free_matrix_1d(double* matrix) {
    free(matrix);
}

void initialize_matrix_zeros_1d(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = 0.0;
    }
}

void initialize_identity_matrix_1d(double* matrix, int dim) {
    initialize_matrix_zeros_1d(matrix, dim, dim);
    for (int i = 0; i < dim; ++i) {
        matrix[i * dim + i] = 1.0; // Диагональные элементы равны 1
    }
}

// Пример инициализации матрицы A (например, A_ij = i + j + 1)
void initialize_matrix_A_1d(double* matrix, int dim) {
    // srand(time(NULL)); // Для случайных чисел, если используется rand()
    // Убедимся, что dim соответствует ожидаемому (3 для этого примера)
    if (dim == 3) {
        matrix[0 * dim + 0] = 0.1; matrix[0 * dim + 1] = 0.4; matrix[0 * dim + 2] = 0.2;
        matrix[1 * dim + 0] = 0.3; matrix[1 * dim + 1] = 0.0; matrix[1 * dim + 2] = 0.5;
        matrix[2 * dim + 0] = 0.6; matrix[2 * dim + 1] = 0.2; matrix[2 * dim + 2] = 0.1;
    } else {
        // Общая инициализация, если DIM не 3 (хотя в задании он 3)
        // Можно оставить ваш предыдущий метод или сообщить об ошибке
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                matrix[i * dim + j] = (double)(i * dim + j + 1.0); // Пример: 1, 2, ..., dim*dim
            }
        }
        // Удаленный блок:
        // if (world_rank == 0) { 
        //     // fprintf(stderr, "Предупреждение: initialize_matrix_A_1d вызвана с dim=%d, но ожидалась 3 для фиксированной матрицы.\n", dim);
        // }
    }
}


void print_matrix_1d(const double* matrix, int rows, int cols, const char* title) {
    printf("%s\n", title);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%8.3f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// C = A * B
void matrix_multiply_1d(const double* A, const double* B, double* C, int dim) {
    initialize_matrix_zeros_1d(C, dim, dim); // Важно: инициализировать C нулями перед умножением
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            for (int k_mult = 0; k_mult < dim; ++k_mult) {
                C[i * dim + j] += A[i * dim + k_mult] * B[k_mult * dim + j];
            }
        }
    }
}

// A_accumulator = A_accumulator + B_term
void matrix_add_1d(double* A_accumulator, const double* B_term, int dim) {
    for (int i = 0; i < dim * dim; ++i) {
        A_accumulator[i] += B_term[i];
    }
}

// C = A / scalar
void matrix_scalar_divide_1d(const double* A, double scalar, double* C, int dim) {
    if (scalar == 0.0) {
        // Эта проверка здесь для полноты, но в данном алгоритме k начинается с 1, так что деления на 0 не будет.
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            fprintf(stderr, "Ошибка: деление на ноль в matrix_scalar_divide_1d.\n");
        }
        // Можно добавить MPI_Abort или другую обработку ошибки
    }
    for (int i = 0; i < dim * dim; ++i) {
        C[i] = A[i] / scalar;
    }
}

void copy_matrix_1d(const double* src, double* dest, int dim) {
    for (int i = 0; i < dim * dim; ++i) {
        dest[i] = src[i];
    }
}