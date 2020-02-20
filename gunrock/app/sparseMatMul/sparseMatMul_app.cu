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
#include <gunrock/app/sparseMatMul/sparseMatMul_enactor.cuh>
#include <gunrock/app/sparseMatMul/sparseMatMul_test.cuh>

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

namespace gunrock {
namespace app {
namespace sparseMatMul {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));

  GUARD_CU(parameters.Use<std::string>(
      "inx", util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::REQUIRED_PARAMETER,
      "", "input file name to feature matrix", __FILE__, __LINE__
  ));
//  GUARD_CU(parameters.Use<std::string>(
//      "inw", util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
//      "", "input file name to weight matrix", __FILE__, __LINE__
//  ));
  GUARD_CU(parameters.Use<int>(
      "hidden_dim", util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0, "hidden dimension of weight matrix", __FILE__, __LINE__
  ));

//  GUARD_CU(parameters.Use<int>(
//      "dim", util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
//      "", "input file name to feature matrix", __FILE__, __LINE__
//  ));

  return retval;
}


}
}
}

template <typename ValueT>
class sprMul {
  typedef typename gunrock::app::TestGraph<int, int, ValueT,
      gunrock::graph::HAS_EDGE_VALUES |
      gunrock::graph::HAS_CSR> GraphT;
  typedef gunrock::app::sparseMatMul::Problem<GraphT> ProblemT;
  typedef gunrock::app::sparseMatMul::Enactor<ProblemT> EnactorT;
  typedef typename GraphT::SizeT SizeT;
  typedef gunrock::util::Parameters Parameters;
  typedef gunrock::util::Array1D<SizeT, ValueT> Array1D;
  ProblemT *problem;
  EnactorT *enactor;
  gunrock::util::Array1D<SizeT, ValueT> *W, *W_grad, *out, *out_grad;

  GraphT readFeature(Parameters &parameters, std::ifstream& svmlight_file,
      int& dim, int& n_rows, int& nnz) {
    std::vector<int> indptr, indices;
    std::vector<ValueT> feature_val;
    n_rows = 0, nnz = 0;
    indptr.push_back(0);

    int max_idx = 0, max_label = 0;
    while(true) {
      std::string line;
      getline(svmlight_file, line);
      if (svmlight_file.eof()) break;
      indptr.push_back(indptr.back());
      std::istringstream ss(line);

      int label = -1;
      ss >> label;
//      labels.push_back(label);
      if (ss.fail()) continue;
      max_label = std::max(max_label, label);

      while (true) {
        std::string kv;
        ss >> kv;
        if(ss.fail()) break;
        std::istringstream kv_ss(kv);

        int k;
        float v;
        char col;
        kv_ss >> k >> col >> v;

        feature_val.push_back(v);
        indices.push_back(k);
        indptr.back() += 1;
        max_idx = std::max(max_idx, k);
      }
    }
    n_rows = indptr.size() - 1;
    nnz = indices.size();
    dim = max_idx + 1;
//    gcnParams->output_dim = max_label + 1;
    GraphT graph;
    // Assign pointers into gunrock graph format
    graph.CsrT::Allocate(n_rows, nnz, gunrock::util::HOST);
    graph.CsrT::row_offsets.SetPointer(indptr.data(), n_rows + 1, gunrock::util::HOST);
    graph.CsrT::column_indices.SetPointer(indices.data(), nnz, gunrock::util::HOST);
    graph.CsrT::edge_values.SetPointer(feature_val.data(), nnz, gunrock::util::HOST);

//  graph.CsrT::row_offsets.Print();
//  graph.CsrT::column_indices.Print();
//  graph.CsrT::edge_values.Print();

//  graph.Display();
    graph.nodes = n_rows;
    graph.edges = nnz;
    gunrock::graphio::LoadGraph(parameters, graph);
    return graph;
  }
public:
  sprMul(Parameters &parameters, std::string X_file, const int hid_dim) {
    problem = new ProblemT(parameters);
    enactor = new EnactorT();

    // reading in sparse matrix
    int nnz, nrows, in_dim;
    auto graph =
        readFeature(parameters, std::ifstream(X_file), in_dim, nrows, nnz);

    // init W and W_grad
    W = new Array1D("W1");
    W_grad = new Array1D ("W1_grad");
    W.Allocate(in_dim * hid_dim);
    W.Allocate(in_dim * hid_dim);

    // init out and out_grad
    out = new Array1D("out");
    out_grad = new Array1D("out_grad");
    out.Allocate(nnz * hid_dim);
    out.Allocate(nnz * hid_dim);

    curandGenerator_t gen;
    curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen,
        std::chrono::system_clock::now().time_since_epoch().count());
    curandGenerateUniformDouble(gen, W.GetPointer(util::DEVICE), in_dim * hid_dim);

    problem->Init(graph, in_dim, hid_dim);
    enactor->Init(problem);
  }

  cudaError_t forward() {
    cudaError_t retval = cudaSuccess;

    problem->Reset(W, out, 1);
    enactor->Reset();
    enactor->enact();
  }

  cudaError_t backward() {
    problem->Reset(out_grad, W_grad, 0);
    enactor->Reset();
    enactor->enact();
  }
};

template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double sparseMatMul(gunrock::util::Parameters &parameters, GraphT &graph, const int dim,
                    const int out_dim, ValueT *in, ValueT *out) {
  typedef typename GraphT::VertexT VertexT;
  typedef gunrock::app::sparseMatMul::Problem<GraphT> ProblemT;
  typedef gunrock::app::sparseMatMul::Enactor<ProblemT> EnactorT;
  gunrock::util::CpuTimer cpu_timer;
  gunrock::util::Location target = gunrock::util::DEVICE;
  double total_time = 0;
  if (parameters.UseDefault("quiet")) parameters.Set("quiet", true);

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  problem.Init(graph, dim, out_dim, in, target);
  enactor.Init(problem, target);

  problem.Reset(in);
  enactor.Reset();

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
double sparseMatMul(gunrock::util::Parameters &parameters, const SizeT n_rows,
    const SizeT nnz, SizeT *row_offsets, VertexT *col_indices, ValueT *vals,
    const int dim, const int outdim, ValueT *b, ValueT *c) {
  typedef typename gunrock::app::TestGraph<VertexT, SizeT, ValueT,
                                           gunrock::graph::HAS_EDGE_VALUES |
                                               gunrock::graph::HAS_CSR>
      GraphT;
  typedef typename GraphT::CsrT CsrT;

  // Setup parameters
//  gunrock::util::Parameters parameters("sparseMatMul");
//  gunrock::graphio::UseParameters(parameters);
//  gunrock::app::sparseMatMul::UseParameters(parameters);
//  gunrock::app::UseParameters_test(parameters);
//  parameters.Parse_CommandLine(0, NULL);
//  parameters.Set("graph-type", "by-pass");
//
//  bool quiet = parameters.Get<bool>("quiet");
  GraphT graph;
  // Assign pointers into gunrock graph format
  graph.CsrT::Allocate(n_rows, nnz, gunrock::util::HOST);
  graph.CsrT::row_offsets.SetPointer(row_offsets, n_rows + 1, gunrock::util::HOST);
  graph.CsrT::column_indices.SetPointer(col_indices, nnz, gunrock::util::HOST);
  graph.CsrT::edge_values.SetPointer(vals, nnz, gunrock::util::HOST);

//  graph.CsrT::row_offsets.Print();
//  graph.CsrT::column_indices.Print();
//  graph.CsrT::edge_values.Print();

//  graph.Display();
  graph.nodes = n_rows;
  graph.edges = nnz;
  gunrock::graphio::LoadGraph(parameters, graph);

  // Run the gcn_graphsum
  double elapsed_time = sparseMatMul(parameters, graph, dim, outdim, b, c);

  // Cleanup
  graph.Release();

  return elapsed_time;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
