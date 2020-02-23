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
#include <gunrock/app/CrossEntropyLoss/CrossEntropyLoss_enactor.cuh>
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
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));

  GUARD_CU(parameters.Use<int>(
      "num_classes",
      util::OPTIONAL_PARAMETER | util::SINGLE_VALUE | util::OPTIONAL_ARGUMENT,
      10, "number of classes per node", __FILE__, __LINE__
  ));

  GUARD_CU(parameters.Use<int>(
      "num_nodes",
      util::OPTIONAL_PARAMETER | util::SINGLE_VALUE | util::OPTIONAL_ARGUMENT,
      1000, "number of nodes", __FILE__, __LINE__
  ));

  return retval;
}

}
}
}

template <typename GraphT, typename ValueT>
class CrossEntropyLoss {
  typedef gunrock::app::CrossEntropyLoss::Problem<GraphT> ProblemT;
  typedef gunrock::app::CrossEntropyLoss::Enactor<ProblemT> EnactorT;
  typedef gunrock::util::Array1D<int, ValueT> Array1D;
  typedef gunrock::util::Parameters Parameters;

  ProblemT *problem;
  EnactorT *enactor;
  Parameters *para;
  Array1D *logits, *grad;
  gunrock::util::Array1D<int, int> *truth;
  int num_nodes, num_classes;
  ValueT loss;
  GraphT *graph;
public:
  CrossEntropyLoss(gunrock::util::Parameters &parameters, GraphT &_graph,
      const int _num_nodes, const int _num_classes, Array1D *_logits,
      gunrock::util::Array1D<int, int> *_ground_truth, Array1D *_grad) :
      para(&parameters), graph(&_graph), num_nodes(_num_nodes),
      num_classes(_num_classes), logits(_logits), grad(_grad), loss(0),
      truth(_ground_truth) {
    problem = new ProblemT(parameters);
    enactor = new EnactorT();
    problem->Init(graph, num_nodes, num_classes, logits, _ground_truth);
    enactor->Init(*problem);
  }

  void forward(bool train = 1) {
    problem->Reset();
    enactor->Reset();
    enactor->Enact();
  }

};

template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double CrossEntropyLoss(gunrock::util::Parameters &parameters, GraphT &graph, const int num_nodes,
                        const int num_classes, ValueT *logits, int *ground_truth, ValueT *grad, ValueT &loss) {
  typedef typename GraphT::VertexT VertexT;
  typedef gunrock::app::CrossEntropyLoss::Problem<GraphT> ProblemT;
  typedef gunrock::app::CrossEntropyLoss::Enactor<ProblemT> EnactorT;
  gunrock::util::CpuTimer cpu_timer;
  gunrock::util::Location target = gunrock::util::DEVICE;
  double total_time = 0;
//  if (parameters.UseDefault("quiet")) parameters.Set("quiet", true);

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  problem.Init(graph, num_nodes, num_classes, logits, ground_truth);
  enactor.Init(problem, target);

  problem.Reset();
  enactor.Reset();

  cpu_timer.Start();
  enactor.Enact();
  cpu_timer.Stop();

  total_time += cpu_timer.ElapsedMillis();
  problem.Extract(grad, &loss);

  enactor.Release(target);
  problem.Release(target);

  return total_time;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
