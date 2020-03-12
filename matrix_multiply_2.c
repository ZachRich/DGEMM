#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <x86intrin.h>
#include <immintrin.h>

#define UNROLL (4)
#define ROWLEN 2048
#define MSIZE [ROWLEN][ROWLEN]

// Compile with: gcc -mavx2 -avx matrix_multiply_2.c


void mm(double a[ROWLEN][ROWLEN], double b[ROWLEN][ROWLEN], double c[ROWLEN][ROWLEN]);
void dgemm(int n, double* A, double* B, double* C);
void dgemm_SIMD(int n, double* A, double* B, double* C);
void dgemm_intrins_SIMD_pipelined(int n, double* A, double* B, double* C);


int main(int argc, char *argv[]) {
    // seed random generator
    srand(2020);
    
    clock_t start, end;
    double cpu_time_used;

    double* a;
    double* b;
    double* c;
    
    a = malloc(ROWLEN * ROWLEN * sizeof(double));
    b = malloc(ROWLEN * ROWLEN * sizeof(double));
    c = malloc(ROWLEN * ROWLEN * sizeof(double));

    /* initialize matricies */

    for(int i=0; i< ROWLEN; i++) {
        for(int j=0; j< ROWLEN; j++) {
            a[i+j*ROWLEN] = (double) rand()/UINT_MAX;
            b[i+j*ROWLEN] = (double) rand()/UINT_MAX;
            c[i+j*ROWLEN] = 0;
        }
    }
    
    
   start = clock(); //Start mm
    printf("Start Matrix Multiply");
    mm(a, b, c);
    printf("End Matrix Multiply");
    end = clock();
    cpu_time_used = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("the elapsed CPU time is %lf\n", cpu_time_used);
    printf("Clocks per sec: %d\n", CLOCKS_PER_SEC);
    
    printf("\n");
    
    start = clock(); //Start DGEMM
    printf("Start DGEMM");
    dgemm(ROWLEN, a, b, c);
    printf("End DGEMM");
    end = clock();
    cpu_time_used = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("the elapsed CPU time is %lf\n", cpu_time_used);
    printf("Clocks per sec: %d\n", CLOCKS_PER_SEC);
    
     printf("\n");
    
    start = clock(); //Start DGEMM
       printf("Start DGEMM_SIMD");
       dgemm_SIMD(ROWLEN, a, b, c);
       printf("End DGEMM_SIMD");
       end = clock();
       cpu_time_used = ((double)(end - start))/CLOCKS_PER_SEC;
       printf("the elapsed CPU time is %lf\n", cpu_time_used);
       printf("Clocks per sec: %d\n", CLOCKS_PER_SEC);
    
     printf("\n");
    
    start = clock(); //Start DGEMM
       printf("Start DGEMM_intrins_SIMD_Pipelined");
       dgemm_intrins_SIMD_pipelined(ROWLEN, a, b, c);
       printf("End DGEMM_intrins_SIMD_Pipelined");
       end = clock();
       cpu_time_used = ((double)(end - start))/CLOCKS_PER_SEC;
       printf("the elapsed CPU time is %lf\n", cpu_time_used);
       printf("Clocks per sec: %d\n", CLOCKS_PER_SEC);
    
     printf("\n");
    
    for (int i = 0; i <  2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("c[%d][%d] = %lf\n", i,j, c[i+j*ROWLEN]);
        }
    }
    return 0;
}


void mm(double a[ROWLEN][ROWLEN], double b[ROWLEN][ROWLEN], double c[ROWLEN][ROWLEN]) {

 int i,j,k;
    for (i = 0; i != ROWLEN; i++) {
        for (j = 0; j != ROWLEN; j++) {
            for (k = 0; k != ROWLEN; k++) {
                c[i][j] = c[i][j] + a[i][k] * b[k][j];
            }
        }
    }
}

 void dgemm (int n, double* A, double* B, double* C)
{
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
        {
            double cij = C[i+j*n]; /* cij = C[i][j] */
            for( int k = 0; k < n; k++ ){
                cij += A[i+k*n] * B[k+j*n]; /* cij += A[i][k]*B[k][j] */
            }
            C[i+j*n] = cij; /* C[i][j] = cij */
        }
    }
 }

void dgemm_SIMD(int n, double* A, double* B, double* C) {
 for ( int i = 0; i < n; i+=4 )
  for ( int j = 0; j < n; j++ ) {
    __m256d c0 = _mm256_load_pd(C+i+j*n); /* c0 = C[i][j] */
    for( int k = 0; k < n; k++ ) {
      c0 = _mm256_add_pd(c0, /* c0 += A[i][k]*B[k][j] */
        _mm256_mul_pd(_mm256_load_pd(A+i+k*n),
        _mm256_broadcast_sd(B+k+j*n)));
    _mm256_store_pd(C+i+j*n, c0); /* C[i][j] = c0 */
  }
}
}


void dgemm_intrins_SIMD_pipelined(int n, double* A, double* B, double* C) {
    
     for ( int i = 0; i < n; i+=UNROLL*4 )
      for ( int j = 0; j < n; j++ ) {
       __m256d c[4];
     for ( int x = 0; x < UNROLL; x++ )
     c[x] = _mm256_load_pd(C+i+x*4+j*n);

     for( int k = 0; k < n; k++ )
     {
     __m256d b = _mm256_broadcast_sd(B+k+j*n);
     for (int x = 0; x < UNROLL; x++)
     c[x] = _mm256_add_pd(c[x],
    _mm256_mul_pd(_mm256_load_pd(A+n*k+x*4+i), b));
     }

     for ( int x = 0; x < UNROLL; x++ )
     _mm256_store_pd(C+i+x*4+j*n, c[x]);
    }
    
}
