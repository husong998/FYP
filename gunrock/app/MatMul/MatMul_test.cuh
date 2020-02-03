// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * gtc_test.cu
 *
 * @brief Test related functions for SSSP
 */

#pragma once

#include <random>
#include <chrono>

#ifdef BOOST_FOUND
// Boost includes for CPU Dijkstra SSSP reference algorithms
#include <boost/config.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>
#else
#include <queue>
#include <vector>
#include <utility>
#endif

namespace gunrock {
namespace app {
namespace MatMul {

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/


/******************************************************************************
 * SSSP Testing Routines
 *****************************************************************************/

void CPU_Reference(double *a, double *b, double *c, int m, int n, int p) {
  for (int i = 0; i < m * p; i++) {
    c[i] = 0;
  }
#pragma omp parallel for schedule(static)
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
#ifdef SIMD
#pragma omp simd
#endif
      for (int k = 0; k < p; k++)
        c[i * p + k] += a[i * n + j] * b[j * p + k];
    }
}

void rand_array(int m, int n, double *weights) {
  std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
  float range = sqrt(6.0f / (m + n));
#pragma omp parallel for schedule(static)
  for(int i = 0; i < m * n; i++)
    weights[i] = (double(rng()) / rng.max() - 0.5) * range * 2;
}


}  // namespace sssp
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
