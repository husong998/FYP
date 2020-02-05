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

#include <gunrock/app/CrossEntropyLoss/CrossEntropyLoss_test.cuh>

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
namespace CrossEntropyLoss {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
//  GUARD_CU(UseParameters_problem(parameters));
//  GUARD_CU(UseParameters_enactor(parameters));

//  GUARD_CU(parameters.Use<int>(
//      "m", util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::REQUIRED_PARAMETER,
//      0, "number of rows in matrix a", __FILE__, __LINE__
//  ));

  return retval;
}

template <typename SizeT, typename ValueT>
class CrossEntropyLoss {
  util::Array1D<SizeT, ValueT> logits, loss;
  util::Array1D<SizeT, int> ground_truth;
  int num_classes, num_nodes;
public:
  cudaError_t CrossEntropyLoss(ValueT *logits, int *ground_truth,
      ValueT *loss, int num_nodes, int num_classes);
  cudaError_t forward(bool);
  cudaError_t backward();
};

template <typename SizeT, typename ValueT>
cudaError_t CrossEntropyLoss::CrossEntropyLoss(ValueT * _logits,
    int * _ground_truth, ValueT * _loss, int _num_nodes, int _num_classes):
    num_classes(_num_classes), num_nodes(_num_nodes) {
  cudaError_t retval = cudaSuccess;

  // moving logits from HOST to DEVICE
  GUARD_CU(logits.Init(num_nodes * num_classes, util::HOST | util::DEVICE))
  GUARD_CU(logits.SetPointer(_logits, num_classes * num_nodes, util::HOST))
  GUARD_CU(logits.Move(util::HOST, util::DEVICE))

  // moving ground_truth from HOST to DEVICE
  GUARD_CU(ground_truth.Init(num_nodes, util::HOST | util::DEVICE))
  GUARD_CU(ground_truth.SetPointer(_ground_truth, num_nodes, util::HOST))
  GUARD_CU(ground_truth.Move(util::HOST, util::DEVICE))

  // Setting loss
  GUARD_CU(loss.Init(1, util::HOST | util::DEVICE))
  GUARD_CU(loss.SetPointer(_loss, util::HOST))
  // Init loss to be 0
  GUARD_CU(loss.ForEach([]__host__ __device__(ValueT &x) { x = 0; }))
}

}
}
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
