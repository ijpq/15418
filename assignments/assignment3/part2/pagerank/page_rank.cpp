#include "page_rank.h"

/*
 * pageRank-- 
 *
 * Computes page rank on a distributed graph g
 * 
 * Per-vertex scores for all vertices *owned by this node* (not all
 * vertices in the graph) should be placed in `solution` upon
 * completion.
 */
void pageRank(DistGraph &g, double* solution, double damping, double convergence) {

    // TODO FOR 15-418/618 STUDENTS:

    // Implement the distributed page rank algorithm here. This is
    // very similar to what you implemnted in Part 1, except in this
    // case, the graph g is distributed across cluster nodes.

    // Each node in the cluster is only aware of the outgoing edge
    // topology for the vertices it "owns".  The cluster nodes will
    // need to coordinate to determine what information.
 
    // note: we give you starter code below to initialize scores for
    // ALL VERTICES in the graph, but feel free to modify as desired.
    // Keep in mind the `solution` array returned to the caller should
    // only have scores for the local vertices
    int totalVertices = g.total_vertices();
    double equal_prob = 1.0/totalVertices;

    int vertices_per_process = g.vertices_per_process;

    std::vector<double> score_curr(totalVertices);
    std::vector<double> score_next(g.vertices_per_process);

    // initialize per-vertex scores
    #pragma omp parallel for
    for (Vertex i = 0; i < totalVertices; i++) {
        score_curr[i] = equal_prob;
    }

    bool converged = false;

    /*

      Repeating basic pagerank pseudocode here for your convenience
      (same as for part 1 of this assignment)

    while (!converged) {

        // compute score_new[vi] for all vertices belonging to this process
        score_new[vi] = sum over all vertices vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
        score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / totalVertices;

        score_new[vi] += sum over all nodes vj with no outgoing edges
                          { damping * score_old[vj] / totalVertices }

        // compute how much per-node scores have changed
        // quit once algorithm has converged

        global_diff = sum over all vertices vi { abs(score_new[vi] - score_old[vi]) };
        converged = (global_diff < convergence)

        // Note that here, some communication between all the nodes is necessary
        // so that all nodes have the same copy of old scores before beginning the 
        // next iteration. You should be careful to make sure that any data you send 
        // is received before you delete or modify the buffers you are sending.

    }

    // Fill in solution with the scores of the vertices belonging to this node.

    */
}
