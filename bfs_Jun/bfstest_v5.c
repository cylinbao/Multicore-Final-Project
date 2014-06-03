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
#include <pthread.h>
#define VISITED 1
#define UNVISITED 0
#define VTXNUM 60000000
#define EDGENUM 90000000

int* push_back(int*, int*, int*);

/* global state */
struct timespec  start_time;                                 
struct timespec  end_time;  

int* vtx[VTXNUM + 1];
int vector_pos[VTXNUM + 1];
int thread_idx[VTXNUM + 1];
int level[VTXNUM + 1];
int VISITED_CHECK[VTXNUM + 1];
int *q;
int * LocalSize_array;
int * LocalOffset_array;
int q_size;
int lvl;
pthread_t* thread_array;
pthread_barrier_t bar;
pthread_mutex_t mutex;

int nv, ne = 0;

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

void* bfs_thread(void* arg)
{
	int idx = *(int*)arg;
	int i;
	int nbr;
	int local_q[10]; 
	int local_size_q = 0;
	int vertex = q[idx];
	for(i = 0; i < vector_pos[vertex]; i++){
		nbr = vtx[vertex][i];
//		printf("The nbr of Vertex(%d): %d\n", vertex, nbr);
//		pthread_mutex_lock(&mutex);
		if(VISITED_CHECK[nbr] == UNVISITED){
			VISITED_CHECK[nbr] = VISITED;
			level[nbr] = lvl;
			local_q[local_size_q] = nbr;
			local_size_q++;
		}
//		pthread_mutex_unlock(&mutex);
	}
	LocalSize_array[idx] = local_size_q;
	pthread_barrier_wait(&bar);
//	printf("TEST BREAKPOINT 1\n");
	if(idx == 0){
//		printf("Thread Index: %d\n", idx);
//		printf("Vertex: %d\n", vertex);
		int global_mem_size = LocalSize_array[0];
		LocalOffset_array[0] = 0;
//		printf("TEST BREAKPOINT 2\n");
		for(i = 1; i < q_size; i ++){
			global_mem_size+=LocalSize_array[i];
			LocalOffset_array[i] = LocalSize_array[i - 1] + LocalOffset_array[i - 1];
		}
//		printf("TEST BREAKPOINT 3\n");
		q_size = global_mem_size;
//		printf("global_mem_size: %d\n", global_mem_size);
		q = (int*) realloc(NULL, q_size*sizeof(int));
	}
//	printf("TEST BREAKPOINT 4: global_mem_size: %d\n", q_size);
	pthread_barrier_wait(&bar);
	int local_offset_q = LocalOffset_array[idx];
	for(i = 0; i < local_size_q; i++)
		q[local_offset_q + i] = local_q[i];
//	printf("TEST BREAKPOINT 5\n");
}
void* bfs_large_thread(void* arg)
{
	int idx = *(int*)arg;
	int i, j, iteration;
	int nbr;
	int local_q[3000]; 
	int local_size_q = 0;
	int vertex ;
	if(idx == q_size/1024) iteration = q_size % 1024;
	else iteration = 1024;
	for(i = 0; i < iteration; i++){
		vertex = q[i + 1024*idx];
		for(j = 0; j < vector_pos[vertex]; j++){
			nbr = vtx[vertex][j];
			if(VISITED_CHECK[nbr] == UNVISITED){
				VISITED_CHECK[nbr] = VISITED;
				level[nbr] = lvl;
				local_q[local_size_q] = nbr;
				local_size_q++;
			}
		}
//	printf("TEST BREAKPOINT 1\n");
	}
//	printf("TEST BREAKPOINT 2\n");
	LocalSize_array[idx] = local_size_q;
	pthread_barrier_wait(&bar);
	if(idx == 0){
//	printf("TEST BREAKPOINT 3\n");
		int global_mem_size = LocalSize_array[0];
		LocalOffset_array[0] = 0;
//		printf("TEST BREAKPOINT 4\n");
		for(i = 1; i < q_size/1024 + 1; i ++){
			global_mem_size+=LocalSize_array[i];
			LocalOffset_array[i] = LocalSize_array[i - 1] + LocalOffset_array[i - 1];
		}
//		printf("TEST BREAKPOINT 5\n");
		q_size = global_mem_size;
		q = (int*) realloc(NULL, q_size*sizeof(int));
//		printf("TEST BREAKPOINT 6(global_mem_size): %d %d\n", q_size, global_mem_size);
	}
	pthread_barrier_wait(&bar);
	int local_offset_q = LocalOffset_array[idx];
//	printf("TEST BREAKPOINT 7(idx: %d, local_size: %d, offset: %d): \n", idx,local_size_q, local_offset_q);
	pthread_barrier_wait(&bar);
	for(i = 0; i < local_size_q; i++)
		q[local_offset_q + i] = local_q[i];
//	printf("TEST BREAKPOINT 8\n");
}
void* bfs_huge_thread(void* arg)
{
	int idx = *(int*)arg;
	int i, j, iteration;
	int nbr;
	int local_q[10000]; 
	int local_size_q = 0;
	int vertex ;
	if(idx == 1023) iteration = q_size/1024 + q_size % 1024;
	else iteration = q_size/1024; 
	for(i = 0; i < iteration; i++){
		vertex = q[i + idx*(q_size/1024)];
		for(j = 0; j < vector_pos[vertex]; j++){
			nbr = vtx[vertex][j];
		//	pthread_mutex_lock(&mutex);
			if(VISITED_CHECK[nbr] == UNVISITED){
				VISITED_CHECK[nbr] = VISITED;
				level[nbr] = lvl;
				local_q[local_size_q] = nbr;
				local_size_q++;
			}
		//	pthread_mutex_unlock(&mutex);
		}
//	printf("TEST BREAKPOINT 1\n");
	}
//	printf("TEST BREAKPOINT 2\n");
	LocalSize_array[idx] = local_size_q;
	pthread_barrier_wait(&bar);
	if(idx == 0){
//	printf("TEST BREAKPOINT 3\n");
		int global_mem_size = LocalSize_array[0];
		LocalOffset_array[0] = 0;
//		printf("TEST BREAKPOINT 4\n");
		for(i = 1; i < 1024 + 1; i ++){
			global_mem_size+=LocalSize_array[i];
			LocalOffset_array[i] = LocalSize_array[i - 1] + LocalOffset_array[i - 1];
		}
//		printf("TEST BREAKPOINT 5\n");
		q_size = global_mem_size;
		q = (int*) realloc(NULL, q_size*sizeof(int));
//		printf("TEST BREAKPOINT 6(global_mem_size): %d %d\n", q_size, global_mem_size);
	}
	pthread_barrier_wait(&bar);
	int local_offset_q = LocalOffset_array[idx];
//	printf("TEST BREAKPOINT 7(idx: %d, local_size: %d, offset: %d): \n", idx,local_size_q, local_offset_q);
	pthread_barrier_wait(&bar);
	for(i = 0; i < local_size_q; i++)
		q[local_offset_q + i] = local_q[i];
//	printf("TEST BREAKPOINT 8\n");
}
	
void bfs()
{
	int i, j, tmp_size;
	q_size = 1;
	q = (int*)malloc(sizeof(int));
	q[0] = 1;
	thread_array = (pthread_t*)malloc(q_size*sizeof(pthread_t));
	LocalSize_array  = (int*) malloc(q_size*sizeof(int));
	LocalOffset_array  = (int*) malloc(q_size*sizeof(int));
	pthread_barrier_init(&bar, NULL, q_size);
	pthread_mutex_init(&mutex, NULL);
	level[0] = -1;
	level[1] = 0;
	for(i = 0; i < VTXNUM + 1; i++)
		thread_idx[i] = i;
	thread_idx[0] = 0;
	
	lvl = 1;
	while(q_size != 0){
		tmp_size = q_size;
		if(tmp_size < 3072){
			thread_array = (pthread_t*)malloc(q_size*sizeof(pthread_t));
			LocalSize_array = (int*) realloc(NULL, q_size*sizeof(int));
			LocalOffset_array = (int*) realloc(NULL, q_size*sizeof(int));
			pthread_barrier_init(&bar, NULL, q_size);
			for(i = 0; i < tmp_size; i++)
				pthread_create(thread_array + i, NULL, bfs_thread, (void*)&thread_idx[i]);
		
			for(i = 0; i < tmp_size; i++)
				pthread_join(thread_array[i], NULL);
		}else if(tmp_size < 1048576){
			thread_array = (pthread_t*)malloc((tmp_size/1024 + 1)*sizeof(pthread_t));
			LocalSize_array = (int*) realloc(NULL, (tmp_size/1024 + 1)*sizeof(int));
			LocalOffset_array = (int*) realloc(NULL, (tmp_size/1024 + 1)*sizeof(int));
			pthread_barrier_init(&bar, NULL, tmp_size/1024 + 1);
			for(i = 0; i < tmp_size/1024 + 1; i++)
				pthread_create(thread_array + i, NULL, bfs_large_thread, (void*)&thread_idx[i]);
			for(i = 0; i < tmp_size/1024 + 1; i++)
				pthread_join(thread_array[i], NULL);
		}else{
			thread_array = (pthread_t*)malloc(1024*sizeof(pthread_t));
			LocalSize_array = (int*) realloc(NULL, 1024*sizeof(int));
			LocalOffset_array = (int*) realloc(NULL, 1024*sizeof(int));
			pthread_barrier_init(&bar, NULL, 1024);
			for(i = 0; i < 1024; i++)
				pthread_create(thread_array + i, NULL, bfs_huge_thread, (void*)&thread_idx[i]);
			for(i = 0; i < 1024; i++)
				pthread_join(thread_array[i], NULL);
			
		}
			
	

	//	printf("The Vertex List Queue(size %d): ", q_size);
	//	for(i = 0; i < q_size; i++)
	//		printf("%d ", q[i]);
	//	printf("\n");
		lvl++;
	}

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
	read_edge_list(vtx, vector_pos, level);
	nv++;
	printf("Num of Edges: %d\n", ne);
	printf("Num of Vertex: %d\n", nv);


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


