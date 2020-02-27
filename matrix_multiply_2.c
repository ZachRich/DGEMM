// thanks to Joshua Skootsky
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <x86intrin.h>


#define ROWLEN 32
#define MSIZE [ROWLEN][ROWLEN]

void mm(double a[ROWLEN][ROWLEN], double b[ROWLEN][ROWLEN], double c[ROWLEN][ROWLEN]);
void dgemmOld(int n, double* A, double* B, double* C);
void dgemm(int n, double* A, double* B, double* C);
void dgemm_intrins();


int main(int argc, char *argv[]) {
    // seed random generator
    srand(time(NULL));

    clock_t start, end;
    double cpu_time_used;
    start = clock();

    double a[ROWLEN][ROWLEN];
    double b[ROWLEN][ROWLEN];
    double c[ROWLEN][ROWLEN];

    /* initialize matricies */

    for(int i=0; i< ROWLEN; i++) {
        for(int j=0; j< ROWLEN; j++) {
            a[i][j] = (double) rand()/UINT_MAX;
            b[i][j] = (double) rand()/UINT_MAX;
            c[i][j] = 0;
        }
    }

    mm(a, b, c);
    dgemmOld(ROWLEN, a, b, c);
	  dgemm(ROWLEN, a, b, c);
    dgemm_intrins();

    end = clock();
    cpu_time_used = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("the elapsed CPU time is %lf\n", cpu_time_used);
    printf("Clocks per sec: %d\n", CLOCKS_PER_SEC);

    for (int i = 0; i <  ROWLEN; i++) {
        for (int j = 0; j < 2; j++) {
            printf("c[%d][%d] = %lf\n", i,j, c[i][j] );
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

void dgemmOld (int n, double* A, double* B, double* C) {

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


void dgemm (int n, double* A, double* B, double* C) {
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

void dgemm_intrins() {
}
