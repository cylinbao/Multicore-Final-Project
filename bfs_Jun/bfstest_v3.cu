// BFSTEST : Test breadth-first search in a graph.
// 
// example: cat sample.txt | ./bfstest 1
//
// John R. Gilbert, 17 Feb 20ll

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <cutil_inline.h>
#define VISITED 1
#define UNVISITED 0
#define VTXNUM 10
#define EDGENUM 17 
#define conti 0
#define stop 1

int* push_back(int*, int*, int*);

/* global state */
struct timespec  start_time;                                 
struct timespec  end_time;  

int* vtx[VTXNUM + 1];
int vector_pos[VTXNUM + 1];
int level[VTXNUM + 1];
int VISITED_CHECK[VTXNUM + 1];
int nbr_list[EDGENUM];
int nbr_offset[VTXNUM + 1];


int nv, ne = 0;


int* d_nbr_list;
int* d_nbr_offset;
int* d_level;
int* d_VISITED_CHECK;
int* d_lvl;
int* d_q2;
int* d_vtx_offset;
int* d_vtx_size;
int* d_num_block;
int* d_q2_size;
int* d_count;

int threadsPerBlock = 256;
int blocksPerGrid;


unsigned int seed = 0x12345678;
unsigned int myrand(unsigned int *seed, unsigned int input) {  
	*seed = (*seed << 13) ^ (*seed >> 15) + input + 0xa174de3;
	return *seed;
};

void sig_check(int nv, int* level) {    
	int i;
	unsigned int sig = 0x123456;

	for(i = 0; i < nv; i++)
	{    
		myrand(&sig, level[i]);    
	}           

	printf("Computed check sum signature:0x%08x\n", sig);
	if(sig == 0x18169857)
		printf("Result check of sample.txt by signature successful!!\n");
	else if(sig == 0xef872cf0)
		printf("Result check of TEST1 by signature successful!!\n");
	else if(sig == 0xe61d1d00) 
		printf("Result check of TEST2 by signature successful!!\n");
	else if(sig == 0x29c12a44)
		printf("Result check of TEST3 by signature successful!!\n");
	else
		printf("Result check by signature failed!!\n");
}

__device__ void enqueue_local(int* index, int nbr, int* q_local)
{
	int old_index = atomicAdd(index, 1);
	q_local[old_index] = nbr;
}

__device__ void global_barrier(int private_num_block, int* count)
{
	__syncthreads();
	if(threadIdx.x == 0){
		atomicAdd(count, 1);
		while(*count < private_num_block){
			;
		}
	}
	__syncthreads();
}


///////////test kernel function//////////////
__global__ void test(int* global_mem_size)
{
	*global_mem_size = 200;
}




__global__ void bfs_kernel(int* q2, int* q2_size, int* vtx_offset, int* vtx_size, int* nbr_list,int* level, int* VISITED_CHECK, int* num_block, int* count)
{
	__shared__ int q_local[1024];
	__shared__ int index;
	int private_num_block = (*num_block);
	if(threadIdx.x == 0)index = 0;
	int i, j= 0;
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	int offset = vtx_offset[idx];
	for(i = 0; i < vtx_size[idx]; i++){
		int nbr = nbr_list[offset + i];
		int old_status = atomicExch(&VISITED_CHECK[nbr], VISITED);
		if(old_status == UNVISITED){
			//enqueue_local(&index, nbr, q_local);

			//////////TEST///////////////
			q_local[j] = nbr;
			j++;
			//////////////////////////
		}
	}
//	global_barrier(private_num_block, count);
	
	if(threadIdx.x == 0)
		atomicAdd(q2_size, j);
	
//	global_barrier(private_num_block, count);
	
	if(threadIdx.x == 0){
		for(i = 0; i < *q2_size; i++)
			q2[i] = q_local[i];
	}
	
}


void read_edge_list (int** vtx, int* vector_pos, int* level) {
	int max_edges = 100000000;
	int nedges, nr, t, h, max;
	nedges = 0;
	nr = scanf("%i %i",&h,&t);
	if(t > h)	nv = t;
	else	nv = h;
	while (nr == 2) {
		if (nedges >= max_edges) {
			printf("Limit of %d edges exceeded.\n",max_edges);
			exit(1);
		}
		vtx[h] = push_back(vtx[h], &t, &vector_pos[h]);
		level[h] = -1;
		level[t] = -1;
		ne++;
		if(t > h)	max = t;
		else	max = h;
		if(max > nv)	nv = max;
		nr = scanf("%i %i",&h,&t);
	}
}

void init_nbr_list(int** vtx, int* vector_pos, int* nbr_list, int* nbr_offset)
{
	int i, j, e;
	e = 0;
	nbr_offset[0] = 0;
	for(j = 0; j < vector_pos[0]; j++){
			nbr_list[e] = vtx[i][j];
			e++;
	}
	for(i = 1; i < VTXNUM; i++){
		nbr_offset[i] = vector_pos[i - 1] + nbr_offset[i - 1];
		for(j = 0; j < vector_pos[i]; j++){
			nbr_list[e] = vtx[i][j];
			e++;
		}
		//free(vector[i]);
		//free(vtx[i]);
	}
	//free(vtx);
}

void bfs()
{
	int i, q1_size, q2_size;
	int init_value = 0;
	int* q_tmp;
	int* vtx_offset;
	int* vtx_size;
	cutilSafeCall(cudaMalloc((void**)&d_q2, EDGENUM*sizeof(int)));
	cutilSafeCall(cudaMalloc((void**)&d_nbr_list, EDGENUM*sizeof(int)));
	cutilSafeCall(cudaMalloc((void**)&d_level, (VTXNUM + 1)*sizeof(int)));
	cutilSafeCall(cudaMalloc((void**)&d_VISITED_CHECK, (VTXNUM + 1)*sizeof(int)));
	cutilSafeCall(cudaMalloc((void**)&d_q2_size, sizeof(int)));
	cutilSafeCall(cudaMalloc((void**)&d_count, sizeof(int)));
	cutilSafeCall(cudaMemcpy(d_nbr_list, nbr_list, (nv + 1)*sizeof(int), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_level, level, (nv + 1)*sizeof(int), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_VISITED_CHECK, VISITED_CHECK, (nv + 1)*sizeof(int), cudaMemcpyHostToDevice));


//////////////////////////FOR LOOP////////////////////////////

	cutilSafeCall(cudaMemcpy(d_count, &init_value, sizeof(int), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_q2_size, &init_value, sizeof(int), cudaMemcpyHostToDevice));
	q1_size = 1;
	q_tmp = (int*)realloc(NULL, q1_size*sizeof(int));
	vtx_offset = (int*)realloc(NULL, q1_size*sizeof(int));
	vtx_size = (int*)realloc(NULL, q1_size*sizeof(int));
	for(i = 0; i < q1_size; i++){
		q_tmp[i] = 1;
		vtx_offset[i] = nbr_offset[q_tmp[i]];
		vtx_size[i] = vector_pos[q_tmp[i]];
	}
	cutilSafeCall(cudaMalloc((void**)&d_vtx_offset, q1_size*sizeof(int)));
	cutilSafeCall(cudaMalloc((void**)&d_vtx_size, q1_size*sizeof(int)));
	cutilSafeCall(cudaMemcpy(d_vtx_offset, vtx_offset, q1_size*sizeof(int), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_vtx_size, vtx_size, q1_size*sizeof(int), cudaMemcpyHostToDevice));

	if(q1_size < 256){
		threadsPerBlock = q1_size;
		blocksPerGrid = 1;
	}
	else{
		threadsPerBlock = 256;
		blocksPerGrid = q1_size/threadsPerBlock;
	}
	cutilSafeCall(cudaMalloc((void**)&d_num_block, sizeof(int)));
	cutilSafeCall(cudaMemcpy(d_num_block, &blocksPerGrid, sizeof(int), cudaMemcpyHostToDevice));
	bfs_kernel<<<threadsPerBlock, blocksPerGrid>>>(d_q2, d_q2_size, d_vtx_offset, d_vtx_size, d_nbr_list, d_level, d_VISITED_CHECK, d_num_block, d_count);
	cutilCheckMsg("kernel launch failure");




///////test/////////////////////////////////////
/*	int* global_mem_size;
	cutilSafeCall(cudaMalloc((void**)&global_mem_size, sizeof(int)));
	cutilSafeCall(cudaMemcpy(global_mem_size, &q1_size, sizeof(int), cudaMemcpyHostToDevice));

	test<<<1, 1>>>(global_mem_size);
	cutilCheckMsg("kernel launch failure");

	cutilSafeCall(cudaMemcpy(&q2_size, global_mem_size, sizeof(int), cudaMemcpyDeviceToHost));*/
//////////////////////////////////////



	cutilSafeCall(cudaMemcpy(&q2_size, d_q2_size, sizeof(int), cudaMemcpyDeviceToHost));
	printf("q2_size: %d\n", q2_size);
	
	q_tmp = (int*)realloc(NULL, q2_size*sizeof(int));

	cutilSafeCall(cudaMemcpy(q_tmp, d_q2, q2_size*sizeof(int), cudaMemcpyDeviceToHost));


	////////////////////////TEST RESULT///////////////////////
	printf("The vtx of q2: ");
	for(i = 0; i < q2_size ; i++)
		printf("%d ", q_tmp[i]);
	printf("\n");

}


int main (int argc, char* argv[]) {
	int startvtx;
	int i /*j*/;
	/*if (argc == 2) {
		startvtx = atoi (argv[1]);
	} else {
		printf("usage:   bfstest <startvtx> < <edgelistfile>\n");
		printf("example: cat sample.txt | ./bfstest 1\n");
		exit(1);
	}*/
	startvtx = 1;

	//int** vtx = (int**)malloc((VTXNUM + 1)*sizeof(int*));
	for(i = 0; i <= VTXNUM; i++){
		if(vtx[i] == NULL)
			vtx[i] = (int*)malloc(sizeof(int));
	}
	//int* nbr_list = (int*)malloc(EDGENUM*sizeof(int));
	//int* vector_pos = (int*)malloc((VTXNUM + 1)*sizeof(int));
	//int* level = (int*)malloc((VTXNUM + 1)*sizeof(int));
	read_edge_list(vtx, vector_pos, level);
	nv++;
	printf("Num of Edges: %d\n", ne);
	printf("Num of Vertex: %d\n", nv);
	printf("Num of Vertex[1]'s link: %d\n", vector_pos[1]);


	//Print the Info of eacg vertex//
	/*for(i = 0; i < nv; i++){
		printf("Vertex[%d]: ", i);
		j = 1;
		while(j<=vector_pos[i]){
			printf("%d ", vtx[i][j]);
			j++;
		}
		printf("\nNum of link: %d\n", vector_pos[i]);
	}*/

	//int* nbr_offset = (int*)malloc((VTXNUM + 1)*sizeof(int));
	//int* VISITED_CHECK = (int*)malloc((VTXNUM + 1)*sizeof(int));

	clock_gettime(CLOCK_REALTIME, &start_time); //stdio scanf ended, timer starts  //Don't remove it

	init_nbr_list(vtx, vector_pos, nbr_list, nbr_offset);

	// Print all the neighbors
	for(i = 0; i < 17; i++)
		printf("%d ", nbr_list[i]);
	printf("\n");
	for(i = 0; i < 10; i++)
		printf("%d ", nbr_offset[i]);
	printf("\n");

	bfs();

		//Print the level of each vertex//
	//for(i = 0; i < 10; i++)
	//	printf("The level of Vertex[%d]: %d\n", i, level[i]);
	//for(i = 0; i < 10; i++)
	//	printf("The VISITED of Vertex[%d]: %d\n", i, VISITED_CHECK[i]);


	clock_gettime(CLOCK_REALTIME, &end_time);  //graph construction and bfs completed timer ends  //Don't remove it


	printf("Starting vertex for BFS is %d.\n\n",startvtx);

	//Don't remove it
	printf("s_time.tv_sec:%ld, s_time.tv_nsec:%09ld\n", start_time.tv_sec, start_time.tv_nsec);
	printf("e_time.tv_sec:%ld, e_time.tv_nsec:%09ld\n", end_time.tv_sec, end_time.tv_nsec);
	if(end_time.tv_nsec > start_time.tv_nsec)
	{
		printf("[diff_time:%ld.%09ld sec]\n",
				end_time.tv_sec - start_time.tv_sec,
				end_time.tv_nsec - start_time.tv_nsec);
	}
	else
	{
		printf("[diff_time:%ld.%09ld sec]\n",
				end_time.tv_sec - start_time.tv_sec - 1,
				end_time.tv_nsec - start_time.tv_nsec + 1000*1000*1000);
	}
	sig_check(nv, level);

	return 0;
}


int* push_back(int *array, int* data, int* pos)
{
	if((*pos) == 0){
//		array = (int*)malloc(sizeof(int));
		array[0] = (*data);
		(*pos) = 1;
	}
	else{
		array = (int*)realloc(array, (*pos+1)*sizeof(int));
		array[(*pos)] = (*data);
		(*pos)++;
	}
	return array;
}


