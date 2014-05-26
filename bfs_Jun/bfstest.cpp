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
#include <vector>
#include <queue>
#include <fstream>
#include <iostream>
#define LOCK 1
#define UNLOCK 0

using namespace std;


/* global state */
struct timespec  start_time;                                 
struct timespec  end_time;  

pair<int, vector<int> > vtx[60000001];
int level[60000001];
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
  int nedges, nr, t, h;
  
  nedges = 0;
  nr = scanf("%i %i",&h,&t);
	nv = max(t, h);
  while (nr == 2) {
    if (nedges >= max_edges) {
      printf("Limit of %d edges exceeded.\n",max_edges);
      exit(1);
    }
		vtx[h].first = UNLOCK;
		vtx[h].first = UNLOCK;
		level[h] = -1;
		level[t] = -1;
		vtx[h].second.push_back(t);
		ne++;
		nv = max(nv, max(t, h));
		nr = scanf("%i %i",&h,&t);
  }
}

void bfs()
{
	queue<int> q;
	int lvl = 1	;
	int samelvl;
	int tmp_samelvl = 0;
	level[0] = -1;
	level[1] = 0;
	for(int i = 0; i < vtx[1].second.size(); i++){
			q.push(vtx[1].second[i]);
			vtx[vtx[1].second[i]].first = LOCK;
			level[vtx[1].second[i]] = lvl;
	}
	lvl++;
	samelvl = vtx[1].second.size();
	while(q.size()!=0){
			for(int i = 0; i < vtx[q.front()].second.size(); i++){
				if(vtx[vtx[q.front()].second[i]].first == UNLOCK){
					q.push(vtx[q.front()].second[i]);
					vtx[vtx[q.front()].second[i]].first = LOCK;
					level[vtx[q.front()].second[i]] = lvl;
					tmp_samelvl++;
				}
			}
			q.pop();
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
  int i, v, reached;
	ofstream ofs;
	ofs.open("result");
  if (argc == 2) {
    startvtx = atoi (argv[1]);
  } else {
    printf("usage:   bfstest <startvtx> < <edgelistfile>\n");
    printf("example: cat sample.txt | ./bfstest 1\n");
    exit(1);
  }

	read_edge_list();
	nv++;
	printf("Num of Edges: %d\n", ne);
	printf("Num of Vertex: %d\n", nv);
	
  clock_gettime(CLOCK_REALTIME, &start_time); //stdio scanf ended, timer starts  //Don't remove it

  bfs ();

	for(int i = 0; i <= 100; i++)
		ofs<<"Level of vertex["<<i<<"]: "<<level[i]<<endl;

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
}




