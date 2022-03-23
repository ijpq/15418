#include <cstring>

using Vertex = int;

/**
 * Class that stores a distributed frontier. Each node has world_size arrays, one
 * dedicated to each node. Add populates these arrays locally, and sync merges
 * them so that the local frontier for each node (containing only local vertices)
 * is present on that node.
 */
class DistFrontier {
  public:
    // Maximum number of vertices that a single node's frontier could have
    // at any given point in time
    int max_vertices_per_node;

    // Distributed frontier structure - every node independently produces a new
    // frontier using its local vertices, and places the frontier vertices in the
    // arrays corresponding to the owning nodes for each destination.
    //
    // For example: elements[2] constains all the frontier vertices that are owned 
    // by process 2
    Vertex **elements;
    int **depths;
    int *sizes;

    int world_size;
    int world_rank;

    DistFrontier(int _max_vertices_per_node, int _world_size, int _world_rank);
    ~DistFrontier();

    void clear();
    void add(int owner_rank, Vertex v, int depth);

    int get_local_frontier_size();
    Vertex* get_local_frontier();

    bool is_empty();
};

inline
DistFrontier::DistFrontier(int _max_vertices_per_node, int _world_size,
                           int _world_rank) :
          max_vertices_per_node(_max_vertices_per_node),
          world_size(_world_size),
          world_rank(_world_rank)
{
  elements = new Vertex*[world_size];
  depths = new int*[world_size];
  for (int i = 0; i < world_size; ++i) {
    elements[i] = new Vertex[max_vertices_per_node];
    depths[i] = new int[max_vertices_per_node];
  }
  
  sizes = new int[world_size]();
}

inline
DistFrontier::~DistFrontier() {
  if (elements) {
    for (int i = 0; i < world_rank; ++i) {
      if ( elements[i] ) delete elements[i];
      if ( depths[i] ) delete depths[i];
    }
    
    delete elements;
    if (depths) delete depths;
  }

  if (sizes) delete sizes;
}

inline
void DistFrontier::clear() {
  memset(sizes, 0, world_size * sizeof(int));
}

inline
void DistFrontier::add(int owner_rank, Vertex v, int depth) {
  elements[owner_rank][sizes[owner_rank]] = v;
  depths[owner_rank][sizes[owner_rank]++] = depth;
}

inline
int DistFrontier::get_local_frontier_size() {
  return sizes[world_rank];
}

inline
Vertex* DistFrontier::get_local_frontier() {
  return elements[world_rank];
}

inline
bool DistFrontier::is_empty() {
  // 15-418/618 STUDENT TODO: Implement this function. Should return
  // true if the cluster-wide frontier is zero
  return true;
}

