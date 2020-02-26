// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_pr.cugraphsum
 *
 * @brief Simple test driver program for PageRank.
 */

#include <gunrock/app/graphsum/graphsum_app.cu>
#include <gunrock/app/test_base.cuh>
#include <cstdio>
#include <iostream>

using namespace gunrock;

/******************************************************************************
 * Main
 ******************************************************************************/

template <typename T>
cudaError_t load_graph(util::Parameters &p, T &g) {
  auto retval = cudaSuccess;
  std::vector<int> row_offsets, col_offsets;
  row_offsets.push_back(0);
  int node = 0;
  std::ifstream graph_file(p.Get<std::string>("graph_file"));
  while(true) {
    std::string line;
    getline(graph_file, line);
    if (graph_file.eof()) break;

    // Implicit self connection
    col_offsets.push_back(node);
    row_offsets.push_back(graph_sparse_index.indptr.back() + 1);
    node++;

    std::istringstream ss(line);
    while (true) {
      int neighbor;
      ss >> neighbor;
      if (ss.fail()) break;
      col_offsets.push_back(neighbor);
      row_offsets.back() += 1;
    }
  }
  g.CsrT::Allocate(node, col_offsets.size(), gunrock::util::HOST);
  g.CsrT::row_offsets.SetPointer(row_offsets.data(), row_offsets.size(), gunrock::util::HOST);
  g.CsrT::column_indices.SetPointer(col_offsets.data(), col_offsets.size(), gunrock::util::HOST);
  g.nodes = node;

  return retval;
}

/**
 * @brief Enclosure to the main function
 */
struct main_struct {
  /**
   * @brief the actual main function, after type switching
   * @tparam VertexT    Type of vertex identifier
   * @tparam SizeT      Type of graph size, i.e. type of edge identifier
   * @tparam ValueT     Type of edge values
   * @param  parameters Command line parameters
   * @param  v,s,val    Place holders for type deduction
   * \return cudaError_t error message(s), if any
   */
  template <typename VertexT,  // Use int as the vertex identifier
            typename SizeT,    // Use int as the graph size type
            typename ValueT>   // Use int as the value type
  cudaError_t
  operator()(util::Parameters &parameters, VertexT v, SizeT s, ValueT val) {
    typedef typename app::TestGraph<VertexT, SizeT, ValueT, graph::HAS_CSR>
        GraphT;
    // typedef typename GraphT::CooT CooT;

    cudaError_t retval = cudaSuccess;
    bool quick = parameters.Get<bool>("quick");
    bool quiet = parameters.Get<bool>("quiet");
    parameters.Set("graph-type", "by-pass");

    util::CpuTimer cpu_timer;
    GraphT graph;

    cpu_timer.Start();
    load_graph(parameters, graph);
    GUARD_CU(graphio::LoadGraph(parameters, graph));
    cpu_timer.Stop();
    parameters.Set("load-time", cpu_timer.ElapsedMillis());

    int dim = parameters.Get<int>("dim");
    freopen(parameters.Get<std::string>("in").c_str(), "r", stdin);
//    freopen(parameters.Get<std::string>("out").c_str(), "w", stdout);
    double *in = new double[graph.nodes * dim], *out = new double[graph.nodes * dim];
    util::PrintMsg("size of in: " + std::to_string(graph.nodes * dim));
    for (int i = 0; i < graph.nodes * dim; i++) scanf("%lf", in + i);
    gcn_graphsum(parameters, graph, dim, in, out);
    for (int i = 0; i < graph.nodes; i++) {
      for (int j = 0; j < dim; j++) std::cout << out[i] * dim + j << ' ';
      std::cout << std::endl;
    }
    return retval;
  }
};

int main(int argc, char **argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test graphsum");
  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(app::graphsum::UseParameters(parameters));
  GUARD_CU(app::UseParameters_test(parameters));
  GUARD_CU(parameters.Parse_CommandLine(argc, argv));
  if (parameters.Get<bool>("help")) {
    parameters.Print_Help();
    return cudaSuccess;
  }
  GUARD_CU(parameters.Check_Required());

  return app::Switch_Types<app::VERTEXT_U32B |  // app::VERTEXT_U64B |
                           app::SIZET_U32B |    // app::SIZET_U64B |
                           app::VALUET_F64B |   // app::VALUET_F64B |
                           app::DIRECTED | app::UNDIRECTED>(parameters,
                                                            main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
