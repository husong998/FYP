// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_pr.cu
 *
 * @brief Simple test driver program for PageRank.
 */

#include <gunrock/app/ReLU/ReLU_app.cu>
#include <gunrock/app/test_base.cuh>
#include <cstdio>
#include <iostream>

using namespace gunrock;

/******************************************************************************
 * Main
 ******************************************************************************/

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

    std::ifstream featurefile(parameters.Get<std::string>("inx"), std::ifstream::in);
    int in_dim, nnz, n_rows;
    double *b, *computed;
    std::vector<int> row_offsets, col_offsets;
    std::vector<double> vals;
    readFeature(featurefile, row_offsets, col_offsets, vals, in_dim, n_rows, nnz);

    int hidden_dim = parameters.Get<int>("hidden_dim");
    double *ref_res = new double[n_rows * hidden_dim];
    b = new double[in_dim * hidden_dim];
    app::sparseMatMul::rand_weights(in_dim, hidden_dim, b);
    app::sparseMatMul::CPU_Reference(row_offsets.data(), col_offsets.data(), vals.data(),
        n_rows, b, in_dim, hidden_dim, ref_res);

    computed = new double[n_rows * hidden_dim];
    sparseMatMul(parameters, n_rows, nnz, row_offsets.data(), col_offsets.data(), vals.data(), in_dim,
        hidden_dim, b, computed);
//    util::CompareResults(computed, ref_res, in_dim * hidden_dim);
    const double EPS = 1e-9;
    for (int i = 0; i < n_rows * hidden_dim; i++) {
      if (fabs(computed[i] - ref_res[i]) > EPS) {
        std::cerr << "failed: [" << i << "]: " << "(" << computed[i] << ", "
        << ref_res[i] << ")\n";
      }
    }
    return retval;
  }
};

int main(int argc, char **argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test ReLU");
  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(app::sparseMatMul::UseParameters(parameters));
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
