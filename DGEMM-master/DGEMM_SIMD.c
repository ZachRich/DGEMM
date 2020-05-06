#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>
#include <immintrin.h>
#include "DGEMM.h"


// void DGEMM_SIMD(int N, double *a, double *b, double *c) {
//     // avx operates on 4 doubles in parallel
//     for (int i=0;  i<N;  i+=4) {
//         for (int j=0;  j<N;  j++) {
//             // c0 = c[i][j]
//             __m256d c0 = {0,0,0,0};
//             for (int k=0;  k<N;  k++) {
//                 c0 = _mm256_add_pd(
//                         c0,   // c0 += a[i][k] * b[k][j]
//                         _mm256_mul_pd(
//                             _mm256_load_pd(a+i+k*N),
//                             _mm256_broadcast_sd(b+k+j*N)));
//             }
//             _mm256_store_pd(c+i+j*N, c0); // c[i,j] = c0
//         }
//     }
// }

void DGEMM_SIMD(int n, double* A, double* B, double* C) {

printf("%s\n", "Entered DGEMM_SIMD!");

 for ( int i = 0; i < n; i+=4 ) {

  for ( int j = 0; j < n; j++ ) {

    __m256d c0 = _mm256_load_pd(C+i+j*n); /* c0 = C[i][j] */

    for( int k = 0; k < n; k++ ) {

      /* c0 += A[i][k]*B[k][j] */
      c0 = _mm256_add_pd(c0,_mm256_mul_pd(_mm256_load_pd(A+i+k*n), _mm256_broadcast_sd(B+k+j*n)));
    _mm256_store_pd(C+i+j*n, c0); /* C[i][j] = c0 */
  }
}
}
}
