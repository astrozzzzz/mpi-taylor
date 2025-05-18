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
    double* current_term = allocate_matrix_1d(DIM, DIM); // Текущий член ряда T_k = A^k/k!
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

    // Инициализация current_term единичной матрицей (для члена k=0: A^0/0! = I)
    initialize_identity_matrix_1d(current_term, DIM);

    // --- Итеративное вычисление членов ряда A^k/k! ---
    // current_term хранит T_k = A^k/k!
    // T_0 = I
    // T_k = (T_{k-1} * A) / k  для k >= 1

    // Обработка члена k=0 (единичная матрица I)
    if (0 % world_size == world_rank) {
        matrix_add_1d(local_sum_A, current_term, DIM); // local_sum_A += current_term (который равен I)
    }

    // Цикл для k = 1 до N_TERMS - 1
    for (int k = 1; k < N_TERMS; ++k) {
        // На данном этапе current_term хранит T_{k-1}
        // Вычисляем T_k = (T_{k-1} * A) / k

        // Шаг 1: T_{k-1} * A
        matrix_multiply_1d(current_term, matrix_A, temp_matrix_mult_result, DIM);
        
        // Шаг 2: (T_{k-1} * A) / k
        // Результат записывается обратно в current_term, который теперь будет T_k
        matrix_scalar_divide_1d(temp_matrix_mult_result, (double)k, current_term, DIM); 

        // Если текущий процесс ответственен за этот член k
        if (k % world_size == world_rank) {
            matrix_add_1d(local_sum_A, current_term, DIM); // local_sum_A += current_term (который T_k)
        }
        // Барьер здесь не обязателен, так как каждый процесс независимо вычисляет
        // последовательность T_k, используя свою копию T_{k-1} и общую matrix_A.
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
// ... existing code ...

// ... existing code ...

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