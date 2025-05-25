#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 3
#define N_TERMS 50000000

typedef struct {
    double data[SIZE][SIZE];
} Matrix;


void matrix_print(const char* label, const double matrix[SIZE][SIZE]);
Matrix matrix_multiply(const double matrix1[SIZE][SIZE], const double matrix2[SIZE][SIZE]);
Matrix matrix_add(const double matrix1[SIZE][SIZE], const double matrix2[SIZE][SIZE]);
void matrix_identity(double matrix[SIZE][SIZE]);
Matrix matrix_scalar_multiply(const double matrix[SIZE][SIZE], double scalar);

#endif
