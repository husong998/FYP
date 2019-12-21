// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file graphsum_app.cu
 *
 * @brief gcn graphsum application
 */

#include <gunrock/gunrock.h>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph definations
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/app/app_base.cuh>
#include <gunrock/app/test_base.cuh>

// single-source shortest path includes
#include <gunrock/app/gcn/graphsum/graphsum_enactor.cuh>

/**
 * @brief      graphsum layer of GCN
 *
 * @param      parameters  The parameters
 * @param      graph       The graph
 * @param[in]  dim         dimension of the feature vector
 * @param      in          the input to the graphsum layer
 * @param      out         output matrix
 *
 * @tparam     GraphT      type of the graph
 * @tparam     ValueT      type of the value, double by default
 *
 * @return     time elapsed to execute
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double gcn_graphsum(gunrock::util::Parameters &parameters, GraphT &graph, const int dim,
                    const ValueT *in, ValueT *out) {
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
  problem.Init(graph, dim, target);
  enactor.Init(problem, dim, target);

  problem.Reset(src, target);
  enactor.Reset(src, target);

  cpu_timer.Start();
  enactor.Enact();
  cpu_timer.Stop();

  total_time += cpu_timer.ElapsedMillis();
  problem.Extract(out);

  enactor.Release(target);
  problem.Release(target);

  return total_time;
}

/*
 * @brief      Simple interface take in graph as CSR format
 *
 * @param[in]  num_nodes    Number of veritces in the input graph
 * @param[in]  num_edges    Number of edges in the input graph
 * @param[in]  row_offsets  CSR-formatted graph input row offsets
 * @param[in]  col_indices  CSR-formatted graph input column indices
 * @param[in]  dim          The dimenssion of the feature vector
 * @param      in           The input to graphsum layer
 * @param      out          The output of graphsum layer
 *
 * @tparam     VertexT      type of vertex id, default to int
 *
 * @return     double      Return accumulated elapsed times for all runs
 */
template <typename VertexT = int, typename SizeT = int, typename ValueT = double>
double graphsum(const SizeT num_nodes, const SizeT num_edges,
            const SizeT *row_offsets, const VertexT *col_indices, const int dim,
            ValueT *in, ValueT *out) {
  typedef typename gunrock::app::TestGraph<VertexT, SizeT, GValueT,
                                           gunrock::graph::HAS_EDGE_VALUES |
                                               gunrock::graph::HAS_CSR>
      GraphT;
  typedef typename GraphT::CsrT CsrT;

  // Setup parameters
  gunrock::util::Parameters parameters("gcn_graphsum");
  gunrock::graphio::UseParameters(parameters);
  gunrock::app::gcn::graphsum::UseParameters(parameters);
  gunrock::app::UseParameters_test(parameters);
  parameters.Parse_CommandLine(0, NULL);
  parameters.Set("graph-type", "by-pass");

  bool quiet = parameters.Get<bool>("quiet");
  GraphT graph;
  // Assign pointers into gunrock graph format
  graph.CsrT::Allocate(num_nodes, num_edges, gunrock::util::HOST);
  graph.CsrT::row_offsets.SetPointer(row_offsets, gunrock::util::HOST);
  graph.CsrT::column_indices.SetPointer(col_indices, gunrock::util::HOST);
  graph.FromCsr(graph.csr(), true, quiet);
  gunrock::graphio::LoadGraph(parameters, graph);

  // Run the gcn_graphsum
  double elapsed_time = gcn_graphsum(parameters, graph, in, out);

  // Cleanup
  graph.Release();

  return elapsed_time;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
