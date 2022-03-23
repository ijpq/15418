#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <string>
#include <iostream>

#include "bfs.h"

#define MASTER 0
#define SILENT "silent"
#define NUM_RUNS 5

int is_same(int *a, int *b, int size) {
  for (int i = 0; i < size; i++)
    if (a[i] != b[i]) return i;
  return size;
}

void print_usage() {
  std::cerr << "Usage: ./bfs <graph_type> <vertices-per-node> <max-edges-per-vertex> [silent]\n";
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

      int *ref_depths = new int[graph_ref.vertices_per_process];
      int *sol_depths = new int[graph_ref.vertices_per_process];

      double best_time_ref = std::numeric_limits<double>::max();
      double best_time_sol = std::numeric_limits<double>::max();
      bool failed = false;

      for (int i = 0; i < NUM_RUNS; ++i) {
        memset(ref_depths, 0, graph_ref.vertices_per_process * sizeof(int));
        
        // Ref timing
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time_ref = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);

        bfs_ref(graph_ref, ref_depths);

        MPI_Barrier(MPI_COMM_WORLD);
        double end_time_ref = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);

        if (world_rank == MASTER)
          best_time_ref = std::min((end_time_ref - start_time_ref), best_time_ref);
      }

      DistGraph graph(vertices_per_node,
                      max_edges_per_vertex,
                      type,
                      world_size,
                      world_rank);

      graph.setup();

      for (int i = 0; i < NUM_RUNS; ++i) {
        memset(sol_depths, 0, graph.vertices_per_process * sizeof(int));

        // Soln timing
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time_sol = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);

        bfs(graph, sol_depths);

        MPI_Barrier(MPI_COMM_WORLD);
        double end_time_sol = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);

        if (world_rank == MASTER)
            best_time_sol = std::min((end_time_sol - start_time_sol), best_time_sol);
      }

      // Verify correctness
      int mismatch = is_same(ref_depths, sol_depths, graph.vertices_per_process);

      if (world_rank == MASTER) {
        if (mismatch != graph.vertices_per_process) {
          failed = true;
          int mismatch_vertex = (MASTER * graph.vertices_per_process) + mismatch;
          int mismatch_val = sol_depths[mismatch_vertex];
          int expected_val = ref_depths[mismatch_vertex];
          std::cerr << "Mismatch at vertex " << mismatch_vertex << ": Got " << mismatch_val << ", expected " <<
              expected_val << "\n";
        }
      }

      MPI_Barrier(MPI_COMM_WORLD);
      if (world_rank == MASTER) {
        for (int j = 1; j < world_size; j++) {
          int mismatch_j, mismatch_val, expected_val;
          MPI_Status status;
          MPI_Recv(&mismatch_j, 1, MPI_INT, j, 0, MPI_COMM_WORLD, &status);

          if (mismatch_j != graph.vertices_per_process) {
            MPI_Recv(&mismatch_val, 1, MPI_INT, j, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&expected_val, 1, MPI_INT, j, 0, MPI_COMM_WORLD, &status);
            std::cerr << "Mismatch at vertex " << j * graph.vertices_per_process
              + mismatch_j << ": Got " << mismatch_val << ", expected " <<
              expected_val << "\n";
            failed = true;
          }
        }
      } else {
        MPI_Send(&mismatch, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
        if (mismatch != graph.vertices_per_process) {
          MPI_Send(&sol_depths[mismatch], 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
          MPI_Send(&ref_depths[mismatch], 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
        }
      }

      int failed_false;
      if (world_rank == MASTER) {
        failed_false = -1;
        MPI_Bcast(&failed_false, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
      } else {
        MPI_Bcast(&failed_false, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
      }

      if (!failed) {
        if (world_rank == MASTER)
          std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(5)
            << "ref time: " << best_time_ref << '\n' << "student time: " <<
               best_time_sol << "\n";
      } else {
        std::cout << "Incorrect\n";
      }

      delete ref_depths;
      delete sol_depths;
    
    } else {
      // Not running in silent mode.
      // You can add debug statements etc here without affecting the autograder.

      if (world_rank == MASTER) {
        std::cout << "Number of processes = " << world_size << "\n";
        std::cout << "Starting graph construction\n";
      }

      if (world_rank == MASTER)
        std::cout << "Done!\n";

      DistGraph graph_sol(vertices_per_node,
                          max_edges_per_vertex,
                          type,
                          world_size,
                          world_rank);

      graph_sol.setup();

      int *sol_depths = new int[graph_sol.vertices_per_process];

      double time_sol = 0.0;

      memset(sol_depths, 0, graph_sol.vertices_per_process * sizeof(int));
      MPI_Barrier(MPI_COMM_WORLD);
      double start_time = MPI_Wtime();
      bfs(graph_sol, sol_depths);
      MPI_Barrier(MPI_COMM_WORLD);
      double end_time = MPI_Wtime();
      time_sol = (end_time - start_time);

      if (world_rank == MASTER) {
        std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(5);
        std::cout << time_sol << "sec \n";
      }

      delete sol_depths;

      DistGraphRef graph(vertices_per_node,
                      max_edges_per_vertex,
                      type,
                      world_size,
                      world_rank);

      graph.change_graph_representation();

      // Compare against sequential single-node BFS solution
      int total_vertices = graph.vertices_per_process * world_size;
      int *ref_sol;
      if (world_rank == MASTER)
        ref_sol = new int[total_vertices];
      else
        ref_sol = new int[graph.vertices_per_process];

      // Aggregate distributed solutions into ref_sol
      bfs_ref(graph, ref_sol);
      if (world_rank == MASTER) {
        for (int i = 1; i < world_size; i++) {
          MPI_Status status;
          MPI_Recv(ref_sol + i * graph.vertices_per_process,
                   graph.vertices_per_process, MPI_INT, i, 0, MPI_COMM_WORLD,
                   &status);
        }
      } else {
        MPI_Send(ref_sol, graph.vertices_per_process, MPI_INT, MASTER, 0,
                 MPI_COMM_WORLD);
      }
    }

    // Finalize the MPI environment.
    MPI_Finalize();
    return 0;
}

