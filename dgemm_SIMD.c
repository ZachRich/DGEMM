
#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>
#include <immintrin.h>


void dgemm_SIMD(int n, double* A, double* B, double* C) {
 for ( int i = 0; i < n; i+=4 )
  for ( int j = 0; j < n; j++ ) {
    __m256d c0 = _mm256_load_pd(C+i+j*n); /* c0 = C[i][j] */
    for( int k = 0; k < n; k++ ) {
      c0 = _mm256_add_pd(c0, /* c0 += A[i][k]*B[k][j] */
        _mm256_mul_pd(_mm256_load_pd(A+i+k*n),
        _mm256_broadcast_sd(B+k+j*n)));
    mm256_store_pd(C+i+j*n, c0); /* C[i][j] = c0 */
  }
}
}
