#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <x86intrin.h>
#include <immintrin.h>
#include <time.h>

#include "DGEMM.h"
#include "mm.c"
#include "DGEMM.c"
#include "DGEMM_SIMD.c"
#include "DGEMM_SIMD_PIPELINED.c"

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
printf("Start Matrix Multiply \n");
mm(a, b, c);
printf("End Matrix Multiply \n");
end = clock();
cpu_time_used = ((double)(end - start))/CLOCKS_PER_SEC;
printf("the elapsed CPU time is %lf\n", cpu_time_used);
printf("Clocks per sec: %d\n", CLOCKS_PER_SEC);

printf("\n");

start = clock(); //Start DGEMM
printf("Start DGEMM \n");
DGEMM(ROWLEN,a, b, c);
printf("End DGEMM \n");
end = clock();
cpu_time_used = ((double)(end - start))/CLOCKS_PER_SEC;
printf("the elapsed CPU time is %lf\n", cpu_time_used);
printf("Clocks per sec: %d\n", CLOCKS_PER_SEC);

 printf("\n");

start = clock(); //Start DGEMM
   printf("Start DGEMM_SIMD \n");
   DGEMM_SIMD(ROWLEN, a, b, c);
   printf("End DGEMM_SIMD \n");
   end = clock();
   cpu_time_used = ((double)(end - start))/CLOCKS_PER_SEC;
   printf("the elapsed CPU time is %lf\n", cpu_time_used);
   printf("Clocks per sec: %d\n", CLOCKS_PER_SEC);

 printf("\n");

start = clock(); //Start DGEMM
   printf("Start DGEMM_intrins_SIMD_Pipelined \n");
   DGEMM_SIMD_PIPELINED(ROWLEN, a, b, c);
   printf("End DGEMM_SIMD_Pipelined \n");
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
