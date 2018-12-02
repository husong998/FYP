// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * knn_enactor.cuh
 *
 * @brief knn Problem Enactor
 */

#pragma once

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/oprtr/oprtr.cuh>

#include <gunrock/app/knn/knn_problem.cuh>
#include <gunrock/util/sort_device.cuh>

#define KNN_DEBUG 1

#ifdef KNN_DEBUG
#define debug(a...) printf(a)
#else
#define debug(a...)
#endif

namespace gunrock {
namespace app {
namespace knn {

/**
 * @brief Speciflying parameters for knn Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));

  return retval;
}

/**
 * @brief defination of knn iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct knnIterationLoop : public IterationLoopBase<EnactorT, Use_FullQ | Push> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
  typedef typename EnactorT::Problem::GraphT::GpT GpT;

  typedef IterationLoopBase<EnactorT, Use_FullQ | Push> BaseIterationLoop;

  knnIterationLoop() : BaseIterationLoop() {}

  /**
   * @brief Core computation of knn, one iteration
   * @param[in] peer_ Which GPU peers to work on, 0 means local
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Core(int peer_ = 0) {
    // --
    // Alias variables

    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];

    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];

    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &graph = data_slice.sub_graph[0];
    auto &adj = data_slice.adj;
    auto &frontier = enactor_slice.frontier;
    auto &oprtr_parameters = enactor_slice.oprtr_parameters;
    auto &retval = enactor_stats.retval;
    auto &iteration = enactor_stats.iteration;

    // struct Point()
    auto &srcs = data_slice.srcs;
    auto &keys = data_slice.keys;
    auto &distances = data_slice.distances;

    // K-Nearest Neighbors
    auto &knns = data_slice.knns;
    auto &core_point = data_slice.core_point;
    auto &cluster = data_slice.cluster;

    // Number of KNNs
    auto k = data_slice.k;
    auto eps = data_slice.eps;
    auto min_pts = data_slice.min_pts;

    // Reference Point
    auto ref_src = data_slice.point_x;
    auto ref_dest = data_slice.point_y;

    // CUB Related storage
    auto &cub_temp_storage = data_slice.cub_temp_storage;

    // Sorted arrays
    auto &keys_out = data_slice.keys_out;
    auto &distances_out = data_slice.distances;

    cudaStream_t stream = oprtr_parameters.stream;
    auto target = util::DEVICE;
    util::Array1D<SizeT, VertexT> *null_frontier = NULL;

    // --
    // Define operations

    // advance operation
    auto distance_op =
        [srcs, keys, distances, knns, k, ref_src, ref_dest] __host__ __device__(
            const VertexT &src, VertexT &dest, const SizeT &edge_id,
            const VertexT &input_item, const SizeT &input_pos,
            SizeT &output_pos) -> bool {
      // Calculate distance between src to edge vertex ref: (x,y)
      VertexT distance = (src - ref_src) * (src - ref_src) +
                         (dest - ref_dest) * (dest - ref_dest);

      // struct Point()
      srcs[edge_id] = src;
      keys[edge_id] = edge_id;  // shouldn't this be like this
      distances[edge_id] = distance;

      // <TODO>
      return true;
      // </TODO>
    };

    // SNN density of each point
    auto density_op =
        [graph, adj, srcs, keys, distances_out, knns, k, eps, min_pts,
         core_point, ref_src, ref_dest] __host__
        __device__(const VertexT &src, VertexT &dest, const SizeT &edge_id,
                   const VertexT &input_item, const SizeT &input_pos,
                   SizeT &output_pos) -> bool {
      auto nodes = graph.nodes;
      int snn_density = 0;
      auto src_num_neighbors = graph.CsrT::GetNeighborListLength(src);
      if (src_num_neighbors < k)
          return false;
      for (auto i = 0; i < k; ++i) {
        // chose i nearest neighbor
        auto neighbor = knns[src * k + i];
        // go over neighbors of the nearest neighbor
        auto f_start = graph.CsrT::GetNeighborListOffset(neighbor);
        auto num_neighbors2 = graph.CsrT::GetNeighborListLength(neighbor);
        int num_shared_neighbors = 0;
        for (int same = f_start; same < f_start + num_neighbors2; ++same) {
          if (adj[src * nodes + same] == 1) ++num_shared_neighbors;
        }
        // if src and neighbor share eps or more neighbors then increase snn
        // density
        if (num_shared_neighbors >= eps) ++snn_density;
      }
      if (snn_density >= min_pts) {
        debug("snn density of %d is %d >= %d, it is core point\n", src, snn_density, min_pts);
        core_point[src] = 1;
      }
      // <TODO>
      return;
      // </TODO>
    };

    // Build cluster
    auto cluster_op =
        [graph, adj, srcs, keys, distances_out, knns, k, eps, min_pts,
         core_point, cluster, ref_src, ref_dest] __host__
        __device__(const VertexT &src, VertexT &dest, const SizeT &edge_id,
                   const VertexT &input_item, const SizeT &input_pos,
                   SizeT &output_pos) -> bool {
      if (core_point[src] != 1) return false;
      auto nodes = graph.nodes;
      for (auto i = 0; i < nodes; ++i) {
        if (i == src || core_point[i] != 1) continue;
        // go over neighbors of core point i
        auto f_start = graph.CsrT::GetNeighborListOffset(i);
        auto num_neighbors = graph.CsrT::GetNeighborListOffset(i);
        int num_shared_neighbors = 0;
        for (int same = f_start; same < f_start + num_neighbors; ++same) {
          if (adj[src * nodes + same] == 1) ++num_shared_neighbors;
        }
        // if src and neighbor share eps or more neighbors then they are in the
        // same cluster
        if (num_shared_neighbors >= eps) {
          auto cluster_id = min(cluster[src], cluster[i]);
          cluster[src] = cluster_id;
          cluster[i] = cluster_id;
          printf("cluster[%d] = %d\n", src, cluster[src]);
          printf("cluster[%d] = %d\n", i, cluster[i]);
        }
      }
      return true;
    };

    oprtr_parameters.advance_mode = "ALL_EDGES";

    // Run
    GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
        graph.csr(), null_frontier, null_frontier, oprtr_parameters,
        distance_op));

    // Sort all the distances using CUB
    GUARD_CU(util::cubSegmentedSortPairs(
        cub_temp_storage, distances, distances_out, keys, keys_out, graph.edges,
        graph.nodes, graph.CsrT::row_offsets, 0,
        std::ceil(std::log2(graph.nodes)), stream));

    // Debug 
    GUARD_CU(keys.ForAll(
        [keys_out, keys, distances, distances_out, graph] 
        __host__ __device__(SizeT * k, const SizeT &pos) {
          printf("after sorting:\n");
          for (int i = 0; i < graph.edges; ++i)
            printf(
                "keys[%d] = %d, keys_out[%d] = %d, dist[%d] = %d, dist_out[%d] "
                "= %d\n",
                i, keys[i], i, keys_out[i], keys[i], distances[keys[i]], keys_out[i],
                distances_out[keys_out[i]]);
        },
        1, util::DEVICE, oprtr_parameters.stream));

    // get reverse keys_out array
    GUARD_CU(keys.ForAll(
        [keys_out]
        __host__ __device__ (SizeT *k, const SizeT &pos){
            k[keys_out[pos]] = pos;
        }, graph.edges, util::DEVICE, oprtr_parameters.stream));
    
    // Choose k nearest neighbors for each node
    GUARD_CU(knns.ForAll(
        [graph, k, keys, keys_out, distances_out] 
        __host__ __device__ (SizeT * knns_, const SizeT &pos) {
          // go to first nearest neighbor
          auto e_start = graph.CsrT::GetNeighborListOffset(pos);
          auto num_neighbors = graph.CsrT::GetNeighborListLength(pos);
          int i = 0;
          for (auto e = e_start; e < e_start + num_neighbors && i < k;
               ++e, ++i) {
            auto m = graph.CsrT::GetEdgeDest(keys_out[keys[e]]);
            knns_[k * pos + i] = m;
          }
        },
        graph.nodes, util::DEVICE, oprtr_parameters.stream));

    // Debug 
    GUARD_CU(knns.ForAll(
        [graph, k]
        __host__ __device__(SizeT * knns_, const SizeT &pos) {
          printf("knns:\n");
          for (int i = 0; i < graph.nodes; ++i){
            printf("%d: ", i);
            auto num_neighbors = graph.CsrT::GetNeighborListLength(i);
            if (num_neighbors < k){
                printf(" # neigh is %d < %d\n", num_neighbors, k);
                continue;
            }
            for (int j = 0; j < k; ++j){
                printf("%d ", knns_[i*k + j]);
            }
            printf("\n");
            }
        }, 1, util::DEVICE, oprtr_parameters.stream));

    // Find density of each point and core points
    GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
        graph.csr(), null_frontier, null_frontier, oprtr_parameters,
        density_op));

    // Assign core points to clusters
    GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
        graph.csr(), null_frontier, null_frontier, oprtr_parameters,
        cluster_op));

    // Assign other non-core and non-noise points to clusters
    GUARD_CU(distances_out.ForAll(
        [graph, keys, keys_out, k, knns, cluster, core_point] __host__ __device__(
            VertexT * d, const VertexT &pos) {
          if (core_point[pos] == 1)
            return;
          auto num_neighbors = graph.CsrT::GetNeighborListLength(pos);
          if (num_neighbors < k)
            return;
          auto e_start = graph.CsrT::GetNeighborListOffset(pos);
          for (auto e = e_start; e < e_start + num_neighbors; ++e) {
            auto m = graph.CsrT::GetEdgeDest(keys_out[keys[e]]);
            if (core_point[m] == 1) {
              cluster[pos] = cluster[m];
              printf("cluster[%d] = %d\n", pos, cluster[pos]);
              return;
            }
          }
        },
        graph.nodes, util::DEVICE, oprtr_parameters.stream));

    // Get back the resulted frontier length
    GUARD_CU(frontier.work_progress.GetQueueLength(
        frontier.queue_index, frontier.queue_length, false, stream, true));

    return retval;
  }

  /**
   * @brief Routine to combine received data and local data
   * @tparam NUM_VERTEX_ASSOCIATES Number of data associated with each
   * transmition item, typed VertexT
   * @tparam NUM_VALUE__ASSOCIATES Number of data associated with each
   * transmition item, typed ValueT
   * @param  received_length The numver of transmition items received
   * @param[in] peer_ which peer GPU the data came from
   * \return cudaError_t error message(s), if any
   */
  template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
  cudaError_t ExpandIncoming(SizeT &received_length, int peer_) {
    // ================ INCOMPLETE TEMPLATE - MULTIGPU ====================

    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    // auto iteration = enactor_slice.enactor_stats.iteration;
    // TODO: add problem specific data alias here, e.g.:
    // auto         &distances          =   data_slice.distances;

    auto expand_op = [
                         // TODO: pass data used by the lambda, e.g.:
                         // distances
    ] __host__ __device__(VertexT & key, const SizeT &in_pos,
                          VertexT *vertex_associate_ins,
                          ValueT *value__associate_ins) -> bool {
      // TODO: fill in the lambda to combine received and local data, e.g.:
      // ValueT in_val  = value__associate_ins[in_pos];
      // ValueT old_val = atomicMin(distances + key, in_val);
      // if (old_val <= in_val)
      //     return false;
      return true;
    };

    cudaError_t retval =
        BaseIterationLoop::template ExpandIncomingBase<NUM_VERTEX_ASSOCIATES,
                                                       NUM_VALUE__ASSOCIATES>(
            received_length, peer_, expand_op);
    return retval;
  }

  bool Stop_Condition(int gpu_num = 0) {
    auto it = this->enactor->enactor_slices[0].enactor_stats.iteration;
    if (it > 0)
      return true;
    else
      return false;
  }
};  // end of knnIteration

/**
 * @brief knn enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <typename _Problem, util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor
    : public EnactorBase<
          typename _Problem::GraphT, typename _Problem::GraphT::VertexT,
          typename _Problem::GraphT::ValueT, ARRAY_FLAG, cudaHostRegisterFlag> {
 public:
  typedef _Problem Problem;
  typedef typename Problem::SizeT SizeT;
  typedef typename Problem::VertexT VertexT;
  typedef typename Problem::GraphT GraphT;
  typedef typename GraphT::VertexT LabelT;
  typedef typename GraphT::ValueT ValueT;
  typedef EnactorBase<GraphT, LabelT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
      BaseEnactor;
  typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag> EnactorT;
  typedef knnIterationLoop<EnactorT> IterationT;

  Problem *problem;
  IterationT *iterations;

  /**
   * @brief knn constructor
   */
  Enactor() : BaseEnactor("KNN"), problem(NULL) {
    // <TODO> change according to algorithmic needs
    this->max_num_vertex_associates = 0;
    this->max_num_value__associates = 1;
    // </TODO>
  }

  /**
   * @brief knn destructor
   */
  virtual ~Enactor() { /*Release();*/
  }

  /*
   * @brief Releasing allocated memory space
   * @param target The location to release memory from
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Release(target));
    delete[] iterations;
    iterations = NULL;
    problem = NULL;
    return retval;
  }

  /**
   * @brief Initialize the problem.
   * @param[in] problem The problem object.
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Init(Problem &problem, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    this->problem = &problem;

    // Lazy initialization
    GUARD_CU(BaseEnactor::Init(
        problem, Enactor_None,
        // <TODO> change to how many frontier queues, and their types
        2, NULL,
        // </TODO>
        target, false));
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      auto &enactor_slice = this->enactor_slices[gpu * this->num_gpus + 0];
      auto &graph = problem.sub_graphs[gpu];
      GUARD_CU(enactor_slice.frontier.Allocate(graph.nodes, graph.edges,
                                               this->queue_factors));
    }

    iterations = new IterationT[this->num_gpus];
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      GUARD_CU(iterations[gpu].Init(this, gpu));
    }

    GUARD_CU(this->Init_Threads(
        this, (CUT_THREADROUTINE) & (GunrockThread<EnactorT>)));
    return retval;
  }

  /**
   * @brief one run of knn, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    gunrock::app::Iteration_Loop<
        // <TODO> change to how many {VertexT, ValueT} data need to communicate
        //       per element in the inter-GPU sub-frontiers
        0, 1,
        // </TODO>
        IterationT>(thread_data, iterations[thread_data.thread_num]);
    return cudaSuccess;
  }

  /**
   * @brief Reset enactor
...
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Reset(
      // <TODO> problem specific data if necessary, eg
      VertexT src = 0,
      // </TODO>
      util::Location target = util::DEVICE) {
    typedef typename GraphT::GpT GpT;
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Reset(target));

    // <TODO> Initialize frontiers according to the algorithm:
    // In this case, we add a single `src` to the frontier
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      if ((this->num_gpus == 1) ||
          (gpu == this->problem->org_graph->GpT::partition_table[src])) {
        this->thread_slices[gpu].init_size = 1;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          auto &frontier =
              this->enactor_slices[gpu * this->num_gpus + peer_].frontier;
          frontier.queue_length = (peer_ == 0) ? 1 : 0;
          if (peer_ == 0) {
            GUARD_CU(frontier.V_Q()->ForEach(
                [src] __host__ __device__(VertexT & v) { v = src; }, 1, target,
                0));
          }
        }
      } else {
        this->thread_slices[gpu].init_size = 0;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          this->enactor_slices[gpu * this->num_gpus + peer_]
              .frontier.queue_length = 0;
        }
      }
    }
    // </TODO>

    GUARD_CU(BaseEnactor::Sync());
    return retval;
  }

  /**
   * @brief Enacts a knn computing on the specified graph.
...
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact(
      // <TODO> problem specific data if necessary, eg
      VertexT src = 0
      // </TODO>
  ) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU KNN Done.", this->flag & Debug);
    return retval;
  }
};

}  // namespace knn
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
