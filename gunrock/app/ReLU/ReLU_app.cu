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

#include <gunrock/app/ReLU/ReLU_test.cuh>

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
namespace ReLU {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
//  GUARD_CU(UseParameters_problem(parameters));
//  GUARD_CU(UseParameters_enactor(parameters));

  GUARD_CU(parameters.Use<int>(
      "len", util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::REQUIRED_PARAMETER,
      0, "length of input matrix", __FILE__, __LINE__
  ));

//  GUARD_CU(parameters.Use<int>(
//      "dim", util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
//      "", "input file name to feature matrix", __FILE__, __LINE__
//  ));

  return retval;
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
  double ReLU(gunrock::util::Parameters &parameters, const SizeT len,
              ValueT *in, ValueT *out) {
    gunrock::util::CpuTimer cpu_timer;

    cpu_timer.Start();
    gunrock::util::Array1D<SizeT, ValueT> a("input");
    a.Init(len, gunrock::util::HOST | gunrock::util::DEVICE);
    a.SetPointer(in, len, gunrock::util::HOST);
    a.Move(gunrock::util::HOST, gunrock::util::DEVICE);
//    a.Print();
    a.ForEach([]__host__ __device__(ValueT &x) {
      x = max(0.0, x);
    }, len, gunrock::util::DEVICE);
    a.SetPointer(out, len, gunrock::util::HOST);
    a.Move(gunrock::util::DEVICE, gunrock::util::HOST);
    cpu_timer.Stop();

    return cpu_timer.ElapsedMillis();
  }


}
}
}

template <typename ValueT>
class relu {
  typedef gunrock::util::Array1D<int, ValueT> Array1D;

  Array1D *in, *in_grad;
public:
  relu(Array1D *_in) : in(_in) {
    *in_grad = new Array1D("relu_grad");
    in_grad->Allocate(in->GetSize(), gunrock::util::DEVICE);
  }
  cudaError_t forward() {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(
        in->ForEach([]__host__ __device__(ValueT &x) {
          x= max(0.0, x);
        }, in->GetSize(), gunrock::util::DEVICE
        ))
  }
  cudaError_t backward() {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(
        in_grad->ForEach(*in,
            []__host__ __device__(ValueT &grad, ValueT &val) {
          if (val < 0) grad = 0;
        }, in_grad->GetSize(), gunrock::util::DEVICE
        ))
  }
};

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
