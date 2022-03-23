#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <string>
#include <iostream>

#include "page_rank.h"

#define MASTER 0
#define SILENT "silent"
#define NUM_RUNS 5

#define PageRankDampening 0.3f
#define PageRankConvergence 0.00001f

// Epsilon for approximate float comparisons
#define EPSILON 0.00000000001

int compareApprox(double* ref, double* stu, int length)
{
  for (int i = 0; i < length; i++) {
    if (fabs(ref[i] - stu[i]) > EPSILON) {
      return i;
    }
  }
  return length;
}


void print_usage() {
  std::cerr << "Usage: ./pr_dist <graph_type> <vertices-per-node> <max-edges-per-vertex> [silent]\n";
  std::cerr << "Available graph types:\n";
  std::cerr << "\t-- uniform_random: A uniform random graph\n";
  std::cerr << "\t-- grid: A grid graph\n";
  std::cerr << "\t\t -> NOTE: vertices-per-node and number of processes should be perfect squares\n";
  std::cerr << "\t-- clustered: A clustered graph (more internal than external edges)\n";
}

/**
 * The ONLY thing written to standard output, if the silent option is provided,
 * will be a single line containing one of these two things:
 * -- Incorrect
 * -- <Avg-ref-soln-time> <Avg-student-soln-time>
 *
 * Grading script will fail if anything other than the above is written.
 */
int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if ( argc < 2 ||  argc > 5 ||
        (argc == 5 && std::string(argv[4]).compare(SILENT)) ) {
      if (world_rank == MASTER)
        print_usage();
      MPI_Finalize();
      exit(1);
    }

    std::string graph_type = std::string(argv[1]);
    GraphType type;
    if (!graph_type.compare("uniform_random"))
      type = uniform_random;
    else if (!graph_type.compare("grid"))
      type = grid;
    else if (!graph_type.compare("clustered"))
      type = clustered;
    else {
      if (world_rank == MASTER)
        print_usage();
      MPI_Finalize();
      exit(1);
    }

    int vertices_per_node = atoi(argv[2]);
    int max_edges_per_vertex = atoi(argv[3]);

    // Running in silent mode, for grading
    if (argc > 4) {
      DistGraphRef graph_ref(vertices_per_node,
                      max_edges_per_vertex,
                      type,
                      world_size,
                      world_rank);

      graph_ref.change_graph_representation();


      DistGraph graph(vertices_per_node,
                      max_edges_per_vertex,
                      type,
                      world_size,
                      world_rank);

      graph.setup();

      double* ref_scores = new double[graph_ref.vertices_per_process]; 
      double* sol_scores = new double[graph_ref.vertices_per_process]; 

      double best_time_ref = std::numeric_limits<double>::max();
      double best_time_sol = std::numeric_limits<double>::max();
      bool failed = false;

      for (int i = 0; i < NUM_RUNS; ++i) {
        memset(ref_scores, 0, graph_ref.vertices_per_process * sizeof(double));
        memset(sol_scores, 0, graph_ref.vertices_per_process * sizeof(double));
        
        // Ref timing
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time_ref = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);

	// run reference solution
        pageRank_ref(graph_ref, ref_scores, PageRankDampening, PageRankConvergence);

        MPI_Barrier(MPI_COMM_WORLD);
        double end_time_ref = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);

        if (world_rank == MASTER)
          best_time_ref = std::min(end_time_ref - start_time_ref, best_time_ref);

        // student soln timing
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time_sol = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);

	// run student's code
        pageRank(graph, sol_scores, PageRankDampening, PageRankConvergence);

        MPI_Barrier(MPI_COMM_WORLD);
        double end_time_sol = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);

        if (world_rank == MASTER)
            best_time_sol = std::min((end_time_sol - start_time_sol), best_time_sol);

        int mismatch = compareApprox(ref_scores, sol_scores, graph.vertices_per_process);

        if (world_rank == MASTER) {
          if (mismatch != graph.vertices_per_process) {
            failed = true;
            int mismatch_vertex = (MASTER * graph.vertices_per_process) + mismatch;
            double mismatch_val = sol_scores[mismatch_vertex];
            double expected_val = ref_scores[mismatch_vertex];
            std::cerr << "Mismatch at vertex " << mismatch_vertex << ": Got " << mismatch_val << ", expected " <<
                expected_val << "\n";
          }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (world_rank == MASTER) {
          for (int j = 1; j < world_size; j++) {
            int mismatch_j;
            double mismatch_val, expected_val;
            MPI_Status status;
            MPI_Recv(&mismatch_j, 1, MPI_INT, j, 0, MPI_COMM_WORLD, &status);

            if (mismatch_j != graph.vertices_per_process) {
              MPI_Recv(&mismatch_val, 1, MPI_DOUBLE, j, 0, MPI_COMM_WORLD, &status);
              MPI_Recv(&expected_val, 1, MPI_DOUBLE, j, 0, MPI_COMM_WORLD, &status);
              std::cerr << "Mismatch at vertex " << j * graph.vertices_per_process
                + mismatch_j << ": Got " << mismatch_val << ", expected " <<
                expected_val << "\n";
              failed = true;
            }
          }
        } else {
          MPI_Send(&mismatch, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
          if (mismatch != graph.vertices_per_process) {
            MPI_Send(&sol_scores[mismatch], 1, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
            MPI_Send(&ref_scores[mismatch], 1, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
          }
        }

        int failed_int;
        if (world_rank == MASTER) {
          failed_int = failed;
          MPI_Bcast(&failed_int, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
        } else {
          MPI_Bcast(&failed_int, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
        }

        if (failed_int) break;
      }

      if (!failed) {
        if (world_rank == MASTER) {
          std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(5)
            << "ref time: " << best_time_ref << '\n' << "student time: " <<
               best_time_sol << "\n";
        }
      } else {
        std::cout << "Incorrect\n";
      }

      if (world_rank == MASTER) {
        double score = !failed ? 5 : 0;
        double ratio = (double)best_time_ref/(best_time_sol + 1e-12);
        if (ratio > 0.4 && !failed) {
            if (ratio > 0.8) {
                score += 15;
            } else {
                score += ((ratio - 0.4)/0.4) * 15;
            }
        }
        std::cout << "Score: " << score << std::endl;
      }

      delete ref_scores;
      delete sol_scores;

    } else {
      // Not running in silent mode.
      // You can add debug statements etc here without affecting the autograder.

      if (world_rank == MASTER) {
        std::cout << "Number of processes = " << world_size << "\n";
        std::cout << "Starting graph construction\n";
      }

      DistGraph graph(vertices_per_node,
                      max_edges_per_vertex,
                      type,
                      world_size,
                      world_rank);

      graph.setup();

      if (world_rank == MASTER)
        std::cout << "Done!\n";

      double *sol_scores = new double[graph.vertices_per_process];

      double best_time_sol = std::numeric_limits<double>::max();

      for (int i = 0; i < NUM_RUNS; ++i) {
        memset(sol_scores, 0, graph.vertices_per_process * sizeof(double));
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();
        pageRank(graph, sol_scores, PageRankDampening, PageRankConvergence);
        MPI_Barrier(MPI_COMM_WORLD);
        double end_time = MPI_Wtime();
        best_time_sol = std::min((end_time - start_time), best_time_sol);
      }

      if (world_rank == MASTER)
        std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(5)
          << "student time: " << best_time_sol << "sec \n";

      delete sol_scores;

    }

    // Finalize the MPI environment.
    MPI_Finalize();
    return 0;
}
