#include <omp.h>
#include "matrix_utils.h"

int main(int argc, char *argv[]) {
    double A[SIZE][SIZE] = {
        {0.1, 0.4, 0.2},
        {0.3, 0.0, 0.5},
        {0.6, 0.2, 0.1}
    };
    double global_taylor_sum[SIZE][SIZE];
    for(int i=0; i<SIZE; ++i) for(int j=0; j<SIZE; ++j) global_taylor_sum[i][j] = 0.0;

    long M_jump_size = 0;

    M_jump_size = (long)sqrt((double)N_TERMS);
    if (M_jump_size == 0) M_jump_size = 1;
    
    Matrix A_pow_M_struct;

    if (M_jump_size == 1) {
        for(int r=0; r<SIZE; ++r) for(int c=0; c<SIZE; ++c) A_pow_M_struct.data[r][c] = A[r][c]; // A^1 = A
    } else {
        matrix_identity(A_pow_M_struct.data);
        for (long i = 0; i < M_jump_size; ++i) {
            A_pow_M_struct = matrix_multiply(A_pow_M_struct.data, A);
        }
    }

    double start_time = omp_get_wtime();
    int num_threads_used = 0;

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        if (thread_id == 0) {
            num_threads_used = num_threads;
            printf("Calculating Taylor series for e^A with %ld terms using OpenMP (%d threads).\n", N_TERMS, num_threads);
        }

        double local_taylor_sum_thread[SIZE][SIZE];
        double current_term_thread[SIZE][SIZE];
        double temp_matrix_thread[SIZE][SIZE];
        Matrix current_T_val_struct_thread;
        double my_initial_term_T_km1_thread[SIZE][SIZE];

        for(int i=0; i<SIZE; ++i) for(int j=0; j<SIZE; ++j) local_taylor_sum_thread[i][j] = 0.0;

        long k_start_idx, k_end_idx;
        long num_my_terms;

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
            k_start_idx = 0; k_end_idx = -1;
        }

        long target_k_for_precalc = k_start_idx - 1;

        if (target_k_for_precalc <= 0) {
            matrix_identity(my_initial_term_T_km1_thread);
        } else {
            matrix_identity(current_T_val_struct_thread.data);
            long current_k_val = 0;

            long num_jumps = target_k_for_precalc / M_jump_size;
            long remaining_steps = target_k_for_precalc % M_jump_size;

            for (long j = 0; j < num_jumps; ++j) {
                double scalar_prod_inv = 1.0;
                for (long l = 1; l <= M_jump_size; ++l) {
                    scalar_prod_inv /= (double)(current_k_val + l);
                }
                Matrix temp_prod_struct = matrix_multiply(current_T_val_struct_thread.data, A_pow_M_struct.data);
                current_T_val_struct_thread = matrix_scalar_multiply(temp_prod_struct.data, scalar_prod_inv);
                current_k_val += M_jump_size;
            }

            for (long l = 0; l < remaining_steps; ++l) {
                current_k_val += 1;
                Matrix temp_prod_struct = matrix_multiply(current_T_val_struct_thread.data, A);
                current_T_val_struct_thread = matrix_scalar_multiply(temp_prod_struct.data, 1.0 / (double)current_k_val);
            }
            for(int r=0; r<SIZE; ++r) for(int c=0; c<SIZE; ++c) my_initial_term_T_km1_thread[r][c] = current_T_val_struct_thread.data[r][c];
        }

        for(int r=0; r<SIZE; ++r) for(int c=0; c<SIZE; ++c) current_term_thread[r][c] = my_initial_term_T_km1_thread[r][c];

        if (num_my_terms > 0) {
            for (long k = k_start_idx; k <= k_end_idx; ++k) {
                Matrix term_A_prod_struct = matrix_multiply(current_term_thread, A);
                for (int r = 0; r < SIZE; ++r) for (int c = 0; c < SIZE; ++c) temp_matrix_thread[r][c] = term_A_prod_struct.data[r][c];
                
                Matrix actual_term_k_struct = matrix_scalar_multiply(temp_matrix_thread, 1.0 / (double)k);
                
                Matrix new_sum_struct = matrix_add(local_taylor_sum_thread, actual_term_k_struct.data);
                for (int r = 0; r < SIZE; ++r) for (int c = 0; c < SIZE; ++c) local_taylor_sum_thread[r][c] = new_sum_struct.data[r][c];
                
                for (int r = 0; r < SIZE; ++r) for (int c = 0; c < SIZE; ++c) current_term_thread[r][c] = actual_term_k_struct.data[r][c];
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < SIZE; i++) {
                for (int j = 0; j < SIZE; j++) {
                    global_taylor_sum[i][j] += local_taylor_sum_thread[i][j];
                }
            }
        }
    }

    double temp_identity_matrix[SIZE][SIZE];
    matrix_identity(temp_identity_matrix);
    Matrix final_sum_struct = matrix_add(global_taylor_sum, temp_identity_matrix);
    for (int r = 0; r < SIZE; ++r) for (int c = 0; c < SIZE; ++c) global_taylor_sum[r][c] = final_sum_struct.data[r][c];

    double end_time = omp_get_wtime();

    matrix_print("Matrix A", A);
    matrix_print("Result e^A (Taylor approximation)", global_taylor_sum);
    printf("Time taken: %f seconds\n", end_time - start_time);

    return 0;
}