#include "matrix_utils.h"

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
