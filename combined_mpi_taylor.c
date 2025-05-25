#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define SIZE 3
#define N_TERMS 50000000

typedef struct {
    double data[SIZE][SIZE];
} Matrix;

void matrix_print(const char* label, const double matrix[SIZE][SIZE]);
Matrix matrix_add(const double matrix1[SIZE][SIZE], const double matrix2[SIZE][SIZE]);
Matrix matrix_multiply(const double matrix1[SIZE][SIZE], const double matrix2[SIZE][SIZE]);
void matrix_identity(double matrix[SIZE][SIZE]);
Matrix matrix_scalar_multiply(const double matrix[SIZE][SIZE], double scalar);
Matrix matrix_power(const double base_matrix[SIZE][SIZE], long exp);

void matrix_sum_mpi_op(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype);

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
        MPI_Abort(MPI_COMM_WORLD, 1);
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

void matrix_sum_mpi_op(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype) {
    double (*in_matrix)[SIZE] = (double (*)[SIZE])invec;
    double (*inout_matrix)[SIZE] = (double (*)[SIZE])inoutvec;

    if (*len != SIZE * SIZE) {
        fprintf(stderr, "Error in matrix_sum_mpi_op\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            inout_matrix[i][j] += in_matrix[i][j];
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, num_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double A[SIZE][SIZE] = {
        {0.2, 0.3, 0.5},
        {0.6, 0.1, 0.3},
        {0.2, 0.3, 0.1}
    };

    double global_taylor_sum[SIZE][SIZE];
    double local_taylor_sum[SIZE][SIZE];
    double current_term[SIZE][SIZE];
    double temp_matrix[SIZE][SIZE];

    for(int i=0; i<SIZE; ++i) {
        for(int j=0; j<SIZE; ++j) {
            local_taylor_sum[i][j] = 0.0;
        }
    }

    long k_start_idx, k_end_idx;
    long num_my_terms;

    long base_count = N_TERMS / num_procs;
    long extra_count = N_TERMS % num_procs;

    if (rank < extra_count) {
        k_start_idx = rank * (base_count + 1) + 1;
        num_my_terms = base_count + 1;
    } else {
        k_start_idx = rank * base_count + extra_count + 1;
        num_my_terms = base_count;
    }
    k_end_idx = k_start_idx + num_my_terms - 1;

    if (num_my_terms == 0) {
        k_start_idx = 0;
        k_end_idx = -1;
    }

    double start_time = MPI_Wtime();

    long M_jump_size = (long)sqrt((double)N_TERMS);
    if (M_jump_size == 0) M_jump_size = 1;

    Matrix A_pow_M_struct;

    if (rank == 0) {
        if (M_jump_size == 1) {
            for(int r=0; r<SIZE; ++r) for(int c=0; c<SIZE; ++c) A_pow_M_struct.data[r][c] = A[r][c];
        } else {
            A_pow_M_struct = matrix_power(A, M_jump_size);
        }
    }

    MPI_Bcast(A_pow_M_struct.data, SIZE*SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    Matrix current_T_val_struct;
    double my_initial_term_T_km1[SIZE][SIZE];

    long target_k_for_precalc = k_start_idx - 1;

    if (target_k_for_precalc <= 0) {
        matrix_identity(my_initial_term_T_km1);
    } else {
        matrix_identity(current_T_val_struct.data);
        long current_k_val = 0;

        for (long j = 0; j < target_k_for_precalc / M_jump_size; ++j) {
            double scalar_prod_inv = 1.0;
            for (long l = 1; l <= M_jump_size; ++l) {
                scalar_prod_inv /= (double)(current_k_val + l);
            }
            Matrix temp_prod_struct = matrix_multiply(current_T_val_struct.data, A_pow_M_struct.data);
            current_T_val_struct = matrix_scalar_multiply(temp_prod_struct.data, scalar_prod_inv);
            current_k_val += M_jump_size;
        }

        for (long l = 0; l < target_k_for_precalc % M_jump_size; ++l) {
            current_k_val += 1;
            Matrix temp_prod_struct = matrix_multiply(current_T_val_struct.data, A);
            current_T_val_struct = matrix_scalar_multiply(temp_prod_struct.data, 1.0 / (double)current_k_val);
        }
        for(int r=0; r<SIZE; ++r) for(int c=0; c<SIZE; ++c) my_initial_term_T_km1[r][c] = current_T_val_struct.data[r][c];
    }

    for(int r=0; r<SIZE; ++r) for(int c=0; c<SIZE; ++c) current_term[r][c] = my_initial_term_T_km1[r][c];

    if (num_my_terms > 0) {
        for (long k = k_start_idx; k <= k_end_idx; ++k) {
            Matrix term_A_prod_struct = matrix_multiply(current_term, A);
            for (int r = 0; r < SIZE; ++r) for (int c = 0; c < SIZE; ++c) temp_matrix[r][c] = term_A_prod_struct.data[r][c];
            
            Matrix actual_term_k_struct = matrix_scalar_multiply(temp_matrix, 1.0 / (double)k);
            
            Matrix new_sum_struct = matrix_add(local_taylor_sum, actual_term_k_struct.data);
            for (int r = 0; r < SIZE; ++r) for (int c = 0; c < SIZE; ++c) local_taylor_sum[r][c] = new_sum_struct.data[r][c];
            
            for (int r = 0; r < SIZE; ++r) for (int c = 0; c < SIZE; ++c) current_term[r][c] = actual_term_k_struct.data[r][c];
        }
    }

    MPI_Op matrix_sum_op;
    MPI_Op_create(matrix_sum_mpi_op, 1, &matrix_sum_op);

    MPI_Reduce(local_taylor_sum, global_taylor_sum, SIZE*SIZE, MPI_DOUBLE, matrix_sum_op, 0, MPI_COMM_WORLD);

    MPI_Op_free(&matrix_sum_op);

    if (rank == 0) {
        matrix_identity(temp_matrix);
        Matrix final_sum_struct = matrix_add(global_taylor_sum, temp_matrix);
        for (int r = 0; r < SIZE; ++r) for (int c = 0; c < SIZE; ++c) global_taylor_sum[r][c] = final_sum_struct.data[r][c];
    }

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("============================================================\n");
        printf("Результаты вычисления экспоненты матрицы e^A:\n");
        printf("============================================================\n");
        matrix_print("Исходная матрица A", A);
        matrix_print("Результат e^A (аппроксимация рядом Тейлора)", global_taylor_sum);
        printf("------------------------------------------------------------\n");
        printf("Затраченное время: %f секунд\n", end_time - start_time);
        printf("Количество членов ряда N_TERMS: %ld\n", (long)N_TERMS);
        printf("Размер матрицы SIZE: %d\n", SIZE);
        printf("Количество MPI процессов: %d\n", num_procs);
        printf("============================================================\n");
    }

    MPI_Finalize();
    return 0;
}