#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>
#include <immintrin.h>

void dgemm_SIMD(int n, double* A, double* B, double* C);
