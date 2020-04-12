CC := gcc
CC := ${CC}

all:
	${CC} DGEMM.h -o DGEMM_h
	${CC} mm.c -c -o mm
	${CC} DGEMM.c -c -o DGEMM
	${CC} DGEMM_SIMD.c -mavx2 -c -o DGEMM_SIMD
	${CC} DGEMM_SIMD_PIPELINED.c -mavx2 -c -o DGEMM_SIMD_PIPELINED
	${CC} DGEMM_CACHE_BLOCKING.c -mavx2 -c -o DGEMM_CACHE_BLOCKING
	${CC} DGEMM_CacheBlock_No_Optimizations.c -mavx2 -c -o DGEMM_CacheBlock_No_Optimizations
	${CC} -mavx2 -o main main.c
