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

int* push_back(int*, int*, int*, int*);

/* global state */
struct timespec  start_time;                                 
struct timespec  end_time;  


int VISITED_CHECK[VTXNUM+1];
int* vtx[VTXNUM+1];
int vector_pos[VTXNUM+1];
int vector_size[VTXNUM+1];
int level[VTXNUM+1];
int nv, ne = 0;

/*typedef struct
{
	int* link;
	int idx;
	int q_idx;
	int size;
}frontier;*/


int* frt_idx;
int** frt_link;
int* frt_link_size;
int* frt_q_idx;

int* d_frt_idx;
int** d_frt_link;
int* d_frt_link_size;
int* d_frt_q_idx;

int* d_level;
int* d_VISITED_CHECK;
int* d_lvl;
int* d_queue;

int threadsPerBlock = 256;
int blocksPerGrid;

void frt_remalloc(int frt_size)
{
	frt_idx = (int*)realloc(NULL, frt_size*sizeof(int));
	frt_link = (int**)realloc(NULL, frt_size*sizeof(int*));
	frt_link_size = (int*)realloc(NULL, frt_size*sizeof(int));
	frt_q_idx = (int*)realloc(NULL, frt_size*sizeof(int));
}

void frt_init(int idx, int vtx_idx)
{
	frt_idx[idx] = vtx_idx;
	frt_link[idx] = (int*)realloc(vtx[vtx_idx], vector_pos[vtx_idx]*sizeof(int));
	frt_link_size[idx] = vector_pos[vtx_idx];
	if(idx == 0)
		frt_q_idx[0] = 0;
	else
		frt_q_idx[idx] = frt_q_idx[idx - 1] + frt_link_size[idx - 1];
}

void frt_gpu_memcpy(int frt_size)
{
	int i;
	cutilSafeCall(cudaMalloc((void**)&d_frt_idx, frt_size*sizeof(int)));
	cutilSafeCall(cudaMalloc((void**)&d_frt_link, frt_size*sizeof(int*)));
	cutilSafeCall(cudaMalloc((void**)&d_frt_link_size, frt_size*sizeof(int)));
	cutilSafeCall(cudaMalloc((void**)&d_frt_q_idx, frt_size*sizeof(int)));
	cutilSafeCall(cudaMemcpy(d_frt_idx, frt_idx, frt_size*sizeof(int), cudaMemcpyHostToDevice));
	for(i = 0; i < frt_size; i++)
		cutilSafeCall(cudaMalloc((void**)&frt_link[i], (frt_link_size[i] + 1)*sizeof(int)));
	cutilSafeCall(cudaMemcpy(d_frt_link, frt_link, frt_size*sizeof(int*), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_frt_link_size, frt_link_size, frt_size*sizeof(int), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_frt_q_idx, frt_q_idx, frt_size*sizeof(int), cudaMemcpyHostToDevice));
}

void frt_DevToHost(int frt_size)
{  
	cutilSafeCall(cudaMemcpy(frt_idx, d_frt_idx, frt_size*sizeof(int), cudaMemcpyDeviceToHost));
  cutilSafeCall(cudaMemcpy(frt_link[0], d_frt_link, frt_size*sizeof(int), cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(frt_link_size, d_frt_link_size, frt_size*sizeof(int), cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(frt_q_idx, d_frt_q_idx, frt_size*sizeof(int), cudaMemcpyDeviceToHost));

	
}



__global__ void bfs_kernel(int* frt_idx, int** frt_link, int* frt_link_size, int* frt_q_idx,  int* level, int* VISITED_CHECK, int* lvl, int* queue)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, link;
	for(i = 0; i < frt_link_size[idx]; i++){
//		*link_list = frt_link[idx];
		link = frt_link[idx][i+1];
		if(VISITED_CHECK[link] == 0){
			VISITED_CHECK[link] = 1;
			level[link] = (*lvl);
			queue[frt_q_idx[idx] + i] = link;
		}
	}
	(*lvl)++;
}



unsigned int seed = 0x12345678;
unsigned int myrand(unsigned int *seed, unsigned int input) {  
	*seed = (*seed << 13) ^ (*seed >> 15) + input + 0xa174de3;
	return *seed;
};

void sig_check(int nv) {    
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

void read_edge_list () {
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
		vtx[h] = push_back(vtx[h], &t, &vector_pos[h], &vector_size[h]);
		level[h] = -1;
		level[t] = -1;
		ne++;
		if(t > h)	max = t;
		else	max = h;
		if(max > nv)	nv = max;
		nr = scanf("%i %i",&h,&t);
	}
}

void bfs()
{
	int i, flag;
	int* lvl;
	int frt_size = 1;
	int* queue;
	int queue_size;
	lvl = (int*)malloc(sizeof(int));
	(*lvl) = 1;
	frt_remalloc(frt_size);
	frt_init(0, 1);
	queue_size = frt_q_idx[frt_size - 1] + frt_link_size[frt_size - 1];
	frt_gpu_memcpy(frt_size);
//	frt_DevToHost(frt_size);
	
	VISITED_CHECK[1] = VISITED;
	level[0] = -1;
	level[1] = 0;

	cutilSafeCall(cudaMalloc((void**)&d_level, (VTXNUM + 1)*sizeof(int)));
	cutilSafeCall(cudaMalloc((void**)&d_VISITED_CHECK, (VTXNUM + 1)*sizeof(int)));
	cutilSafeCall(cudaMalloc((void**)&d_lvl, sizeof(int)));

	cutilSafeCall(cudaMemcpy(d_level, level, (nv + 1)*sizeof(int), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_VISITED_CHECK, VISITED_CHECK, (nv + 1)*sizeof(int), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_lvl, lvl, sizeof(int), cudaMemcpyHostToDevice));

	while(true){
		if(frt_size < 256){
			threadsPerBlock = frt_size;
			blocksPerGrid = 1;
		}
		else{
			threadsPerBlock = 256;
			blocksPerGrid = frt_size/threadsPerBlock ;
		}
		queue = (int*)malloc(queue_size*(sizeof(int)));
		cutilSafeCall(cudaMalloc((void**)&d_queue, queue_size*sizeof(int)));

		bfs_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_frt_idx, d_frt_link, d_frt_link_size, d_frt_q_idx, d_level, d_VISITED_CHECK, d_lvl, d_queue);
//	frt_DevToHost(frt_size);
		cutilCheckMsg("kernel launch failure");    

//		cutilSafeCall(cudaMemcpy(level, d_level, (nv + 1)*sizeof(int), cudaMemcpyDeviceToHost));


		cutilSafeCall(cudaMemcpy(queue, d_queue, queue_size*sizeof(int), cudaMemcpyDeviceToHost));
		flag = 1;
		frt_size = 0;
		for(i = 0; i < queue_size; i++){
			if(queue[i]!=0){
				frt_size++;
				flag = 0;
			}
		}
		if(flag)
			break;
		frt_remalloc(frt_size);
		frt_size = 0;
		for(i = 0; i < queue_size; i++){
			if(queue[i]!=0){
				frt_init(frt_size, queue[i]);
				frt_size++;
			}
		}
		queue_size = frt_q_idx[frt_size - 1] + frt_link_size[frt_size - 1];
		frt_gpu_memcpy(frt_size);
	}
	cutilSafeCall(cudaMemcpy(level, d_level, (nv + 1)*sizeof(int), cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(VISITED_CHECK, d_VISITED_CHECK, (nv + 1)*sizeof(int), cudaMemcpyDeviceToHost));
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

	for(i = 0; i < VTXNUM; i++){
		if(vtx[i] == NULL)
			vtx[i] = (int*)malloc(sizeof(int));
	}

	read_edge_list();
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



	clock_gettime(CLOCK_REALTIME, &start_time); //stdio scanf ended, timer starts  //Don't remove it

	bfs();

		//Print the level of each vertex//
	for(i = 0; i < 10; i++)
		printf("The level of Vertex[%d]: %d\n", i, level[i]);
	for(i = 0; i < 10; i++)
		printf("The VISITED of Vertex[%d]: %d\n", i, VISITED_CHECK[i]);


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
	sig_check(nv);

	return 0;
}


int* push_back(int *array, int* data, int* pos, int* size)
{
	if((*size) == 0){
		array = (int*)malloc(2*sizeof(int));
		array[1] = (*data);
		(*pos) = 1;
		(*size) = 2;
	}
	else if((*pos)+1 <= (*size)){
		array[(*pos)+1] = (*data);
		(*pos)++;
	}
	else if((*pos)+1 > (*size)){
		array = (int*)realloc(array, 2*(*size)*sizeof(int));
		array[(*pos)+1] = (*data);
		(*size) = (*size)*2;
		(*pos)++;
	}
	return array;
}


