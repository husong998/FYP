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

#include <gunrock/app/MatMul/MatMul_app.cu>
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
    int m = parameters.Get<int>("m");
    int n = parameters.Get<int>("n");
    int p = parameters.Get<int>("p");

    double *a = new double[m * n], *b = new double[n * p], *c_ref = new double[m * p], *c_calc = new double[m * p];
    app::MatMul::rand_array(m, n, a);
    app::MatMul::rand_array(n, p, b);

    app::MatMul::CPU_Reference(a, b, c_ref, m, n, p);
    app::MatMul::MatMul(parameters, a, b, c_calc, m, n, p);

    const double EPS = 1e-9;
    for (int i = 0; i < m * p; i++) {
//      std::cerr << "in[" << i << "]: " << in[i] << ", " << "out[" << i << "]: " << out[i] << std::endl;
      if (fabs(c_ref[i] - c_calc[i]) > EPS) {
        using namespace std;
        cerr << "faild on index " << i << "; ";
        cerr << "ref[" << i << "]: " << c_ref[i] << ", " << "calc[" << i << "]: " << c_calc[i] << endl;
      }
    }
    return retval;
  }
};

int main(int argc, char **argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test MatMul");
  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(app::MatMul::UseParameters(parameters));
  GUARD_CU(app::UseParameters_test(parameters));
  GUARD_CU(parameters.Parse_CommandLine(argc, argv));
  parameters.Set("graph-type", "by-pass");
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
