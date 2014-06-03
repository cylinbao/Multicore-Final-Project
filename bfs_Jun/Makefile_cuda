CXXFLAG = -O0 -gencode=arch=compute_11,code=\"sm_11,compute_11\"  -gencode=arch=compute_20,code=\"sm_20,compute_20\" --compiler-options -fno-strict-aliasing  -I . -I /usr/local/cuda/include -I ~/NVIDIA_GPU_Computing_SDK/C/common/inc -I ~/NVIDIA_GPU_Computing_SDK/shared/inc -DUNIX
LDFLAG = -fPIC -L /usr/local/cuda/lib -L ~/NVIDIA_GPU_Computing_SDK/C/lib -L ~/NVIDIA_GPU_Computing_SDK/C/common/lib/linux -L ~/NVIDIA_GPU_Computing_SDK/shared/lib -lcudart -L /usr/local/cuda/lib -L ~/NVIDIA_GPU_Computing_SDK/C/lib -L ~/NVIDIA_GPU_Computing_SDK/C/common/lib/linux -L ~/NVIDIA_GPU_Computing_SDK/shared/lib -lcudart -lcutil_i386 -lshrutil_i386 -fopenmp -lrt -lm
SRCS = bfstest.cu
TARGET = bfstest

all:									
	nvcc $(CXXFLAG) -c -o main.o ${SRCS}
	
	g++ -o ${TARGET} *.o  $(LDFLAG) 
			
clean:
	rm -rf *.o main
	rm -rf *.log	
