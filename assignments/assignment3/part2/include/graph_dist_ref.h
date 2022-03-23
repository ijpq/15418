#ifndef __DISTGRAPHREF_DEFINED
#define __DISTGRAPHREF_DEFINED

#include <mpi.h>
#include <stdio.h>
#include <random>
#include <set>
#include <cassert>
#include <cmath>
#include <iostream>

// 75% of the edges for any vertex will be internal edges
#define CLUSTERING_RATIO 0.75f

using Vertex = int;

enum GraphType { uniform_random, rmat, grid, clustered };

struct Edge {
    Vertex src;
    Vertex dest;
};

class DistGraphRef {
public:
    int vertices_per_process;
    int max_edges_per_vertex;
    GraphType type;

    int world_size;
    int world_rank;

    Vertex start_vertex;
    Vertex end_vertex;

    DistGraphRef(int _vertices_per_process, int _max_edges_per_vertex,
              GraphType _type, int _world_size, int _world_rank);

    int get_vertex_owner_rank(Vertex v);
    void generate_graph_uniform();

    bool is_left_edge_vertex(Vertex v, int sqrt_world_size, int sqrt_per_process);
    bool is_right_edge_vertex(Vertex v, int sqrt_world_size, int sqrt_per_process);
    bool is_top_edge_vertex(Vertex v, int sqrt_world_size, int sqrt_per_process);
    bool is_bottom_edge_vertex(Vertex v, int sqrt_world_size, int sqrt_per_process);
    void generate_graph_grid();

    void generate_graph_clustered();
    void get_incoming_edges(const std::vector<std::vector<Edge>> &edge_scatter);
    int total_vertices();

    // Edge.dest should always be local to this node
    std::vector<Edge> in_edges;
    // Edge.src should always be local to this node
    std::vector<Edge> out_edges;

    // TODO: Implement internal representations suitable for doing bfs/pagerank
    // like part 1 from the in_edges and out_edges vectors

    std::vector<std::vector<Vertex>> incoming_edges;
    std::vector<std::vector<Vertex>> outgoing_edges;

    // Compute lists of edges per vertex (incoming/outgoing_edges) from
    // list of edges (in/out_edges)
    void change_graph_representation();
};

#endif
