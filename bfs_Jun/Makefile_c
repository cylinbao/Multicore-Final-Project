CC = gcc
CFLAGS = -g3 -O2
LIBS = -fopenmp -lrt -lm -lpthread
SRCS = bfstest.c
TARGET = bfstest
all:
	gcc ${CFLAGS}  ${LIBS}  ${SRCS} -o ${TARGET} -lrt
clean:
	rm -rf ${TARGET}.out ${TARGET} 
