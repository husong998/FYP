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
namespace ReLU {

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/


/******************************************************************************
 * SSSP Testing Routines
 *****************************************************************************/

void rand_array(int len, double *weights) {
  std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
#pragma omp parallel for schedule(static)
  for(int i = 0; i < len; i++)
    weights[i] = rng() - (double)rng.max() / 2;
}


}  // namespace sssp
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
