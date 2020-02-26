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
#include <gunrock/app/gcn/gcn_enactor.cuh>

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
namespace gcn {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));

  GUARD_CU(parameters.Use<std::string>(
      "in",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::REQUIRED_PARAMETER,
      "invalid",
      "input file name to feature matrix", __FILE__, __LINE__
      ));

  GUARD_CU(parameters.Use<int>(
      "dim",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::REQUIRED_PARAMETER,
      -1,
      "feature vector dimension", __FILE__, __LINE__
      ));

  GUARD_CU(parameters.Use<std::string>(
      "out",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::REQUIRED_PARAMETER,
      "out",
      "output file name", __FILE__, __LINE__
      ));

  return retval;
}


}
}
}

template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double gcn(gunrock::util::Parameters &parameters, GraphT &graph, const int dim,
                    ValueT *in, ValueT *out) {
  typedef typename GraphT::VertexT VertexT;
  typedef gunrock::app::gcn::Problem<GraphT> ProblemT;
  typedef gunrock::app::gcn::Enactor<ProblemT> EnactorT;
  gunrock::util::CpuTimer cpu_timer;
  gunrock::util::Location target = gunrock::util::DEVICE;
  double total_time = 0;
  parameters.Set("quiet", false);

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  problem.Init(graph);
  enactor.Init(problem, target);

  problem.Reset();
  enactor.Reset();

  cpu_timer.Start();
  enactor.Enact();
  cpu_timer.Stop();

  total_time += cpu_timer.ElapsedMillis();
//  problem.Extract(out);

  enactor.Release(target);
  problem.Release(target);

  return total_time;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
