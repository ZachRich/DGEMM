
#include "DGEMM.h"

void mm(double* A, double* B, double* C) {

 int i,j,k;
    for (i = 0; i != ROWLEN; i++) {
        for (j = 0; j != ROWLEN; j++) {
            for (k = 0; k != ROWLEN; k++) {
                //C[i][j] = C[i][j] + A[i][k] * B[k][j];
                 C[i+j*ROWLEN] = A[i+k*ROWLEN] * B[k+j*ROWLEN];
            }
        }
    }
}
