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

#include <gunrock/app/cross_entropy_loss/cross_entropy_loss_app.cu>
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

    int num_classes = parameters.Get<int>("num_classes");
    int num_nodes = parameters.Get<int>("num_nodes");

    // randomizing inputs
    int *ground_truth = new int[num_nodes];
    app::cross_entropy_loss::rand_truth(num_classes, num_nodes, ground_truth);
    double *logits = new double[num_classes * num_nodes];
    app::cross_entropy_loss::rand_logits(num_classes * num_nodes, logits);

    // run CPU_reference as benchmark
    double *ref_grad = new double[num_classes * num_nodes], ref_loss;
    app::cross_entropy_loss::CPU_Reference(num_nodes, num_classes, logits, ground_truth, ref_grad, ref_loss);

    double *cal_grad = new double[num_classes * num_nodes], cal_loss;
    GraphT g; // dummy graph to be passed to gunrock problem struct
    cross_entropy_loss(parameters, g, num_nodes, num_classes, logits, ground_truth, cal_grad, cal_loss);
    return retval;
  }
};

int main(int argc, char **argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test graphsum");
  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(app::cross_entropy_loss::UseParameters(parameters));
  GUARD_CU(app::UseParameters_test(parameters));
  GUARD_CU(parameters.Parse_CommandLine(argc, argv));
  GUARD_CU(parameters.Set("graph-type", "bypass"))
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
