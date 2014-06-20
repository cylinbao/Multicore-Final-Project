CC = gcc
CFLAGS = -O3
LIBS = -lrt -lm -lpthread -fopenmp
SRCS = bfstest_v6.c
TARGET = bfstest
all:
	gcc ${CFLAGS}  ${LIBS}  ${SRCS} -o ${TARGET} -lrt 
clean:
	rm -rf ${TARGET}.out ${TARGET} 
