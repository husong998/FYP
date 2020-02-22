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

#include <gunrock/app/MatMul/MatMul_test.cuh>

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
namespace MatMul {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
//  GUARD_CU(UseParameters_problem(parameters));
//  GUARD_CU(UseParameters_enactor(parameters));

  GUARD_CU(parameters.Use<int>(
      "m", util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::REQUIRED_PARAMETER,
      0, "number of rows in matrix a", __FILE__, __LINE__
  ));

  GUARD_CU(parameters.Use<int>(
      "n", util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::REQUIRED_PARAMETER,
      0, "number of columns in matrix a", __FILE__, __LINE__
  ));

  GUARD_CU(parameters.Use<int>(
      "p", util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::REQUIRED_PARAMETER,
      0, "number of columns in matrix b", __FILE__, __LINE__
  ));

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
  template <typename SizeT = int, typename ValueT = double>
  cudaError_t MatMul(gunrock::util::Parameters &parameters, ValueT *_a, ValueT *_b, ValueT *_c,
      const SizeT m, const SizeT n, const SizeT p) {

    cudaError_t retval = cudaSuccess;
//    gunrock::util::CpuTimer cpu_timer;

//    cpu_timer.Start();
    gunrock::util::Array1D<SizeT, ValueT> c("c"), a("a"), b("b");

    // Initializing a
    GUARD_CU(a.Init(m * n, gunrock::util::DEVICE | gunrock::util::HOST));
    GUARD_CU(a.SetPointer(_a, m * n, gunrock::util::HOST));
    GUARD_CU(a.Move(gunrock::util::HOST, gunrock::util::DEVICE));
//    a.Print();

    // Initializing b
    GUARD_CU(b.Init(n * p, util::DEVICE | util::HOST));
    GUARD_CU(b.SetPointer(_b, n * p, gunrock::util::HOST));
    GUARD_CU(b.Move(util::HOST, util::DEVICE));
//    b.Print();

    // Initializing c
    GUARD_CU(c.Init(m * p, gunrock::util::DEVICE | util::HOST));
    GUARD_CU(c.ForEach([]__host__ __device__(ValueT &x) {
      x = 0;
    }));

    // Calculating matrix multiplication
    GUARD_CU(a.ForAll(
        [b, c, p, n]__host__ __device__(ValueT *a_, const SizeT pos) {
          int i = pos / n, j = pos % n;
          for (int k = 0; k < p; k++) {
            atomicAdd(c + i * p + k, a_[pos] * b[j * p + k]);
//            printf("i: %d\n", i * p + k);
          }
        }, m * n, util::DEVICE
        ));

//    GUARD_CU(c.Print());
    // Extracting results
    GUARD_CU(c.SetPointer(_c, m * p, gunrock::util::HOST));
    GUARD_CU(c.Move(gunrock::util::DEVICE, gunrock::util::HOST));

//    cpu_timer.Stop();

    return retval;
  }


}
}
}

template <typename ValueT>
class matMul {
  typedef gunrock::util::Array1D<int, ValueT> Array1D;
  Array1D *a, *b, *c, *a_grad, *b_grad, *c_grad;
  int n, m, p;
public:
  matMul(Array1D *_a, Array1D *_a_grad, Array1D *_b, Array1D *_b_grad,
      Array1D *_c, Array1D *_c_grad, int _m, int _n, int _p) : a(_a), b(_b),
      c(_c), a_grad(_a_grad), b_grad(_b_grad), c_grad(_c_grad), n(_n), m(_m),
      p(_p) {}

  cudaError_t forward() {
    cudaError_t retval = cudaSuccess;
    assert(a->GetSize() == m * n && b->GetSize() == n * p &&
    c->GetSize() == m * p);

    GUARD_CU(c->ForEach([]__host__ __device__(ValueT &x) {
      x = 0;
    }));

    GUARD_CU(a->ForAll(
        [b, c, p, n]__host__ __device__(ValueT *a_, const int pos) {
          int i = pos / n, j = pos % n;
          for (int k = 0; k < p; k++) {
            atomicAdd(c + i * p + k, a_[pos] * b[j * p + k]);
    //            printf("i: %d\n", i * p + k);
          }
        }, m * n, gunrock::util::DEVICE
        ))

     return retval;
  }

  cudaError_t backward() {
    cudaError_t retval = cudaSuccess;
    assert(a_grad->GetSize() == m * n && b_grad->GetSize() == n * p &&
           c_grad->GetSize() == m * p);

    GUARD_CU(a_grad->ForEach([]__host__ __device__(ValueT &x) {
      x = 0;
    }));

    GUARD_CU(b_grad->ForEach([]__host__ __device__(ValueT &x) {
      x = 0;
    }));

    GUARD_CU(a_grad->ForAll(
        [b_grad, c_grad, p, n]__host__ __device__(ValueT *a_, const int pos) {
          int i = pos / n, j = pos % n;
          for (int k = 0; k < p; k++) {
            atomicAdd(b_grad + j * p + k, a_[pos] * c_grad[i * p + k]);
            //            printf("i: %d\n", i * p + k);
          }
        }, m * n, gunrock::util::DEVICE
        ))

    return retval;
  }
};

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
