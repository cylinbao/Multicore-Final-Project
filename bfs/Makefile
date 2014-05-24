CC = gcc
CFLAGS = -g -O3
LIBS = -fopenmp -lrt -lm -lpthread
SRCS = bfstest.c
TARGET = bfstest
all:
	gcc ${CFLAGS}  ${LIBS}  ${SRCS} -o ${TARGET} -lrt
clean:
	rm -rf ${TARGET}.out ${TARGET} 
