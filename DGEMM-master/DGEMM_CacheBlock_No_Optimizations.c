#define BLOCKSIZE 32


void do_block_No_Optimization(int n, int si, int sj, int sk, double *A, double *B, double *C) {
    for (int i = si; i < si+BLOCKSIZE; ++i)
        for (int j = sj; j < sj+BLOCKSIZE; ++j) {
            double cij = C[i+j*n];/* cij = C[i][j] */

            for( int k = sk; k < sk+BLOCKSIZE; k++ )
                cij += A[i+k*n] * B[k+j*n];/* cij+=A[i][k]*B[k][j] */
            C[i+j*n] = cij;/* C[i][j] = cij */
        }
 }
void dgemm_Cacheblock_No_Optimization(int n, double* A, double* B, double* C) {

    for ( int sj = 0; sj < n; sj += BLOCKSIZE )
        for ( int si = 0; si < n; si += BLOCKSIZE )
            for ( int sk = 0; sk < n; sk += BLOCKSIZE )
                do_block_No_Optimization(n, si, sj, sk, A, B, C);
 }
