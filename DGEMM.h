#ifndef DGEMM_H_
#define DGEMM_H_
#define ROWLEN 128
#define UNROLL (4)

void mm(double* A, double* B, double* C);

void DGEMM(int n, double* A, double* B, double* C);

void DGEMM_SIMD(int n, double* A, double* B, double* C);

void DGEMM_SIMD_PIPELINED(int n, double* A, double* B, double* C);

void do_block(int n, int si, int sj, int sk, double *A, double *B, double *C);

void dgemm_block(int n, double* A, double* B, double* C);

#endif //DGEMM_H_
