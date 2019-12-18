// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file graphsum_app.cu
 *
 * @brief Gunrock graphsum layer in GCN
 */

// <primitive>_app.cuh includes
#include <gunrock/app/app.cuh>

// page-rank includes
#include <gunrock/app/pr/pr_enactor.cuh>
#include <gunrock/app/pr/pr_test.cuh>


template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double gunrock_pagerank(gunrock::util::Parameters &parameters, GraphT &graph,
                        typename GraphT::VertexT **node_ids, ValueT **ranks) {
  typedef typename GraphT::VertexT VertexT;
  typedef gunrock::app::gcn::graphsum::Problem<GraphT> ProblemT;
  typedef gunrock::app::gcn::graphsum::Enactor<ProblemT> EnactorT;

  gunrock::util::CpuTimer cpu_timer;
  gunrock::util::Location target = gunrock::util::DEVICE;
  double total_time = 0;
  if (parameters.UseDefault("quiet")) parameters.Set("quiet", true);

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;

  printf("Init Problem and Enactor for graphsum layer.\n");
  problem.Init(graph, target);
  enactor.Init(problem, target);

  std::vector<VertexT> srcs = parameters.Get<std::vector<VertexT>>("srcs");
  int num_runs = parameters.Get<int>("num-runs");
  int num_srcs = srcs.size();
  for (int run_num = 0; run_num < num_runs; ++run_num) {
    printf("For run_num: %d, Reset problem and enactor and Enact.\n", run_num);
    int src_num = run_num % num_srcs;
    VertexT src = srcs[src_num];
    problem.Reset(src, target);
    enactor.Reset(src, target);

    cpu_timer.Start();
    enactor.Enact(src);
    cpu_timer.Stop();

    total_time += cpu_timer.ElapsedMillis();
    enactor.Extract();
    problem.Extract(node_ids[src_num], ranks[src_num]);
  }

  enactor.Release(target);
  problem.Release(target);
  srcs.clear();
  return total_time;
}

/**
 * Interface for graphsum layer taking in CSR format
 * @tparam VertexT
 * @tparam SizeT
 * @tparam ValueT
 * @param [in]  num_nodes     number of nodes in graph
 * @param [in]  num_edges     number of edges in graph
 * @param [in]  row_offsets   number of nnz entries for each row
 * @param [in]  col_indices   column id for each nnz entry
 * @param [in]  dim           dimension for each node feature
 * @param [in]  feature_in    input feature matrix
 * @param [out] feature_out   output feature matrix
 * @return
 */
template <typename VertexT = int, typename SizeT = int, typename ValueT = float>
double graphsum(const SizeT num_nodes, const SizeT num_edges,
                const SizeT *row_offsets, const VertexT *col_indices, const int dim,
                ValueT *feature_in, ValueT *feature_out) {
  typedef typename gunrock::app::TestGraph<
      VertexT, SizeT, ValueT, gunrock::graph::HAS_COO | gunrock::graph::HAS_CSC>
      GraphT;
  typedef typename gunrock::app::TestGraph<VertexT, SizeT, ValueT,
                                           gunrock::graph::HAS_CSR>
      Graph_CsrT;
  typedef typename Graph_CsrT::CsrT CsrT;

  // Setup parameters
  gunrock::util::Parameters parameters("graphsum");
  gunrock::graphio::UseParameters(parameters);
  gunrock::app::UseParameters_test(parameters);
  parameters.Parse_CommandLine(0, NULL);
  parameters.Set("graph-type", "by-pass");

  bool quiet = parameters.Get<bool>("quiet");

  CsrT csr;
  // Assign pointers into gunrock graph format
  csr.Allocate(num_nodes, num_edges, gunrock::util::HOST);
  csr.row_offsets.SetPointer((int *)row_offsets, num_nodes + 1,
                             gunrock::util::HOST);
  csr.column_indices.SetPointer((int *)col_indices, num_edges,
                                gunrock::util::HOST);
  // csr.Move(gunrock::util::HOST, gunrock::util::DEVICE);

  gunrock::util::Location target = gunrock::util::HOST;

  GraphT graph;
  graph.FromCsr(csr, target, 0, quiet, true);
  csr.Release();
  gunrock::graphio::LoadGraph(parameters, graph);

  // Run the PR
  double elapsed_time = gunrock_pagerank(parameters, graph, node_ids, ranks);

  // Cleanup
  // graph.Release();
  // srcs.clear();

  return elapsed_time;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
