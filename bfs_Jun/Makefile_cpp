CC = g++
CFLAGS = -g -O2
LIBS = -fopenmp -lrt -lm -lpthread
SRCS = bfstest.cpp
TARGET = bfstest
all:
	g++ ${CFLAGS}  ${LIBS}  ${SRCS} -o ${TARGET} -lrt
clean:
	rm -rf ${TARGET}.out ${TARGET} 
