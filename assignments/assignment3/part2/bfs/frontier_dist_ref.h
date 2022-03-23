#include <cstring>

using Vertex = int;

/**
 * Class that stores a distributed frontier. Each node has world_size arrays, one
 * dedicated to each node. Add populates these arrays locally, and sync merges
 * them so that the local frontier for each node (containing only local vertices)
 * is present on that node.
 */
class DistFrontierRef {
  public:
    // Maximum number of vertices that a single node's frontier could have
    // at any given point in time
    int max_vertices_per_node;

    // Distributed frontier structure - every node independently produces a new
    // frontier using its local vertices, and places the frontier vertices in the
    // arrays corresponding to the owning nodes for each destination.
    Vertex **elements;
    int **depths;
    int *sizes;

    int world_size;
    int world_rank;

    DistFrontierRef(int _max_vertices_per_node, int _world_size, int _world_rank);
    ~DistFrontierRef();

    void clear();
    void add(int owner_rank, Vertex v, int depth);

    int get_local_frontier_size();
    Vertex* get_local_frontier();

    bool is_empty();
};
