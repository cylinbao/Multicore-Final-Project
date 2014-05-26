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


int VISITED_CHECK[VTXNUM+1];
int* vtx[VTXNUM+1];
int vector_pos[VTXNUM+1];
int vector_size[VTXNUM+1];
int level[VTXNUM+1];
int q[EDGENUM+1];
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
		VISITED_CHECK[h] = UNVISITED;
		VISITED_CHECK[h] = UNVISITED;
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
	int lvl;
	int samelvl;
	int tmp_samelvl;
	int i, q_front, q_back;
	level[0] = -1;
	level[1] = 0;
	lvl = 1;
	tmp_samelvl = 0;
	for(i = 1; i <= vector_pos[1]; i++){
		q[i] = vtx[1][i];
		VISITED_CHECK[vtx[1][i]] = VISITED;
		level[vtx[1][i]] = lvl;
	}
	lvl++;
	samelvl = vector_pos[1];
	q_front = 1;
	q_back = vector_pos[1];
	while(q_front != q_back){
		for(i = 1; i <= vector_pos[q[q_front]]; i++){
			if(VISITED_CHECK[vtx[q[q_front]][i]] == UNVISITED){
				q[++q_back] = vtx[q[q_front]][i];
				VISITED_CHECK[vtx[q[q_front]][i]] = VISITED;
				level[vtx[q[q_front]][i]] = lvl;
				tmp_samelvl++;
			}
		}
		q_front++;
		samelvl--;
		if(samelvl == 0){
			samelvl = tmp_samelvl;
			tmp_samelvl = 0;
			lvl++;
		}
	}
}



int main (int argc, char* argv[]) {
	int startvtx;
	int i, j, v, reached;
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
	for(i = 0; i <= 100; i++)
		printf("The level of Vertex[%d]: %d\n", i, level[i]);


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
