// thanks to Joshua Skootsky
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <x86intrin.h>
#include "dgemm.h"
#include "dgemm_SIMD.h"


#define ROWLEN 32
#define MSIZE [ROWLEN][ROWLEN]





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


    dgemm(ROWLEN, a, b, c);
	  dgemm_SIMD(ROWLEN, a, b, c);
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
