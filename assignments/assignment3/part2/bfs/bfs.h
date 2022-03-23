#include "graph_dist_ref.h"
#include "frontier_dist_ref.h"
#include "graph_dist.h"
#include "frontier_dist.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

#ifdef DEBUG
#include <cassert>
#endif

// Reference solution
void global_frontier_sync_ref(DistGraphRef &g, DistFrontierRef &frontier, int *depths);

void bfs_step_ref(DistGraphRef &g, int *depths, DistFrontierRef &current_frontier,
              DistFrontierRef &next_frontier, std::set<int> &done_foreign_vertices);

void bfs_ref(DistGraphRef &g, int *depths);

void bfs_ref_sequential(DistGraphRef &g, int *depths);

/**
 * Takes a distributed graph, and a distributed frontier with each node containing
 * world_size independently produced new frontiers, and merges them such that each
 * node contains exactly one frontier with all its local vertices.
 */
void global_frontier_sync(DistGraph &g, DistFrontier &frontier, int *depths);

void bfs_step(DistGraph &g, int *depths, DistFrontier &current_frontier,
              DistFrontier &next_frontier, std::set<int> &done_foreign_vertices);

void bfs(DistGraph &g, int *depths);