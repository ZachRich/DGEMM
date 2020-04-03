#ifndef DGEMM_H_
#define DGEMM_H_
#define ROWLEN 512
#define UNROLL (4)

void mm(double* A, double* B, double* C);

void DGEMM(int n, double* A, double* B, double* C);

void DGEMM_SIMD(int n, double* A, double* B, double* C);

void DGEMM_SIMD_PIPELINED(int n, double* A, double* B, double* C);


#endif //DGEMM_H_
