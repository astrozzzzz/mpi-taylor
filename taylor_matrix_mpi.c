#include <mpi.h>
#include "matrix_utils.h"

void matrix_sum_mpi_op(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype) {
    double (*in_matrix)[SIZE] = (double (*)[SIZE])invec;
    double (*inout_matrix)[SIZE] = (double (*)[SIZE])inoutvec;

    if (*len != SIZE * SIZE) {
        fprintf(stderr, "Error in matrix_sum_mpi_op: len mismatch!\n");
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
        {0.1, 0.4, 0.2},
        {0.3, 0.0, 0.5},
        {0.6, 0.2, 0.1}
    };
    double global_taylor_sum[SIZE][SIZE];
    double local_taylor_sum[SIZE][SIZE];
    double current_term[SIZE][SIZE];
    double temp_matrix[SIZE][SIZE];

    for(int i=0; i<SIZE; ++i) for(int j=0; j<SIZE; ++j) local_taylor_sum[i][j] = 0.0;

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
        k_start_idx = 0; k_end_idx = -1;
    }

    double start_time = MPI_Wtime();

    long M_jump_size = 0;
    M_jump_size = (long)sqrt((double)N_TERMS);
    if (M_jump_size == 0) M_jump_size = 1;

    Matrix A_pow_M_struct;

    if (rank == 0) {
        if (M_jump_size == 1) {
            for(int r=0; r<SIZE; ++r) for(int c=0; c<SIZE; ++c) A_pow_M_struct.data[r][c] = A[r][c]; // A^1 = A
        } else {
            matrix_identity(A_pow_M_struct.data);
            for (long i = 0; i < M_jump_size; ++i) {
                A_pow_M_struct = matrix_multiply(A_pow_M_struct.data, A);
            }
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
        matrix_print("Matrix A", A);
        matrix_print("Result e^A (Taylor approximation)", global_taylor_sum);
        printf("Time taken: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}