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
#define VISITED 1
#define UNVISITED 0
#define VTXNUM 60000000
#define EDGENUM 90000000

int* push_back(int*, int*, int*, int*);

/* global state */
struct timespec  start_time;                                 
struct timespec  end_time;  


int VISIT_CHECK[VTXNUM+1];
int* vtx[VTXNUM+1];
int vector_pos[VTXNUM+1];
int vector_size[VTXNUM+1];
int level[VTXNUM+1];
int queue[EDGENUM+1];
int nv, ne = 0;

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
}



int main (int argc, char* argv[]) {
	int startvtx;
	int i, v, reached;
//	if (argc == 2) {
//		startvtx = atoi (argv[1]);
//	} else {
//		printf("usage:   bfstest <startvtx> < <edgelistfile>\n");
//		printf("example: cat sample.txt | ./bfstest 1\n");
//		exit(1);
//	}
	startvtx = 1;
	for(i = 0; i < VTXNUM + 1; i++){
		vector_pos[i] = 0;
		vector_size[i] = 0;
	}
	read_edge_list();
	nv++;
	printf("Num of Edges: %d\n", ne);
	printf("Num of Vertex: %d\n", nv);

	clock_gettime(CLOCK_REALTIME, &start_time); //stdio scanf ended, timer starts  //Don't remove it

	//bfs();

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
	//sig_check(nv);

	return 0;
}


int* push_back(int *array, int* data, int* pos, int* size)
{
	int* new_array;
	if((*size) == 0){
		array = (int*)malloc(2*sizeof(int));
		array[0] = (*data);
		(*pos) = 1;
		(*size) = 2;
	}
	else if((*pos)+1 <= (*size)){
		array[(*pos)+1] = *data;
		(*pos)++;
	}
	else if((*pos)+1 > (*size)){
		new_array = (int*)malloc(2*(*size));
		memcpy(new_array, array, (*pos));
		free(array); 
		new_array[(*pos)+1] = (*data);
		(*size) = (*size)*2;
		(*pos)++;
		return new_array;
	}
	return array;
}
