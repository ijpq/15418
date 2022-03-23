#include <cstring>
#include <set>
#include <iostream>
#include <vector>
#include <queue>
#include "bfs.h"

#define contains(container, element) \
  (container.find(element) != container.end())

/**
 *
 * global_frontier_sync--
 * 
 * Takes a distributed graph, and a distributed frontier with each node containing
 * world_size independently produced new frontiers, and merges them such that each
 * node holds the subset of the global frontier containing local vertices.
 */
void global_frontier_sync(DistGraph &g, DistFrontier &frontier, int *depths) {


  // TODO 15-418/618 STUDENTS
  //
  // In this function, you should synchronize between all nodes you
  // are using for your computation. This would mean sending and
  // receiving data between nodes in a manner you see fit. Note for
  // those using async sends: you should be careful to make sure that
  // any data you send is received before you delete or modify the
  // buffers you are sending.

  int world_size = g.world_size;
  int world_rank = g.world_rank;
}

/*
 * bfs_step --
 * 
 * Carry out one step of a distributed bfs
 * 
 * depths: current state of depths array for local vertices
 * current_frontier/next_frontier: copies of the distributed frontier structure
 * 
 * NOTE TO STUDENTS: We gave you this function as a stub.  Feel free
 * to change as you please (including the arguments)
 */
void bfs_step(DistGraph &g, int *depths,
	      DistFrontier &current_frontier,
              DistFrontier &next_frontier) {

  int frontier_size = current_frontier.get_local_frontier_size();
  Vertex* local_frontier = current_frontier.get_local_frontier();

  // keep in mind, this node owns the vertices with global ids:
  // g.start_vertex, g.start_vertex+1, g.start_vertex+2, etc...

  // TODO 15-418/618 STUDENTS
  //
  // implement a step of the BFS

}

/*
 * bfs --
 * 
 * Execute a distributed BFS on the distributed graph g
 * 
 * Upon return, depths[i] should be the distance of the i'th local
 * vertex from the BFS root node
 */
void bfs(DistGraph &g, int *depths) {
  DistFrontier current_frontier(g.vertices_per_process, g.world_size,
                                g.world_rank);
  DistFrontier next_frontier(g.vertices_per_process, g.world_size,
                             g.world_rank);

  DistFrontier *cur_front = &current_frontier,
               *next_front = &next_frontier;

  // Initialize all the depths to NOT_VISITED_MARKER.
  // Note: Only storing local vertex depths.
  for (int i = 0; i < g.vertices_per_process; ++i )
    depths[i] = NOT_VISITED_MARKER;

  // Add the root node to the frontier 
  int offset = g.start_vertex;
  if (g.get_vertex_owner_rank(ROOT_NODE_ID) == g.world_rank) {
    current_frontier.add(g.get_vertex_owner_rank(ROOT_NODE_ID), ROOT_NODE_ID, 0);
    depths[ROOT_NODE_ID - offset] = 0;
  }

  while (true) {

    bfs_step(g, depths, *cur_front, *next_front);

    // this is a global empty check, not a local frontier empty check.
    // You will need to implement is_empty() in ../dist_graph.h
    if (next_front->is_empty())
      break;

    // exchange frontier information
    global_frontier_sync(g, *next_front, depths);

    DistFrontier *temp = cur_front;
    cur_front = next_front;
    next_front = temp;
    next_front -> clear();
  }
}

