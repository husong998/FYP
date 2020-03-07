// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * graphsum_problem.cuh
 *
 * @brief GPU Storage management Structure for SSSP Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/sparseMatMul/sparseMatMul_app.cu>
#include <gunrock/app/graphsum/graphsum_app.cu>
#include <gunrock/app/CrossEntropyLoss/CrossEntropyLoss_app.cu>

namespace gunrock {
namespace app {
namespace gcn {
/**
 * @brief Speciflying parameters for graphsum Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(gunrock::app::UseParameters_problem(parameters));
  GUARD_CU(parameters.Use<int>("in_dim",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0, "input_dimension", __FILE__, __LINE__))

  GUARD_CU(parameters.Use<int>("out_dim",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0, "input_dimension", __FILE__, __LINE__))

  GUARD_CU(parameters.Use<int>("hid_dim",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      16, "input_dimension", __FILE__, __LINE__))

  GUARD_CU(parameters.Use<double>("learning_rate",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0.005, "learning rate", __FILE__, __LINE__))

  return retval;
}

struct module {
  virtual ~module() = default;
  virtual void forward(bool) = 0;
  virtual void backward() = 0;
  virtual double GetLoss() {
    return 0;
  }
};

/**
 * @brief Single-Source Shortest Path Problem structure.
 * @tparam _GraphT  Type of the graph
 * @tparam _LabelT  Type of labels used in sssp
 * @tparam _ValueT  Type of per-vertex distance values
 * @tparam _FLAG    Problem flags
 */
template <typename _GraphT, typename _LabelT = typename _GraphT::VertexT,
          typename _ValueT = typename _GraphT::ValueT,
          ProblemFlag _FLAG = Problem_None>
struct Problem : ProblemBase<_GraphT, _FLAG> {
  typedef _GraphT GraphT;
  static const ProblemFlag FLAG = _FLAG;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CsrT CsrT;
  typedef typename GraphT::GpT GpT;
  typedef _LabelT LabelT;
  typedef _ValueT ValueT;

  typedef ProblemBase<GraphT, FLAG> BaseProblem;
  typedef DataSliceBase<GraphT, FLAG> BaseDataSlice;
  typedef util::Parameters Parameters;

  typedef util::Array1D<SizeT, ValueT> Array;
  typedef typename app::TestGraph<VertexT, SizeT, ValueT, graph::HAS_CSR | graph::HAS_EDGE_VALUES>
      SpmatT;

  // Helper structures

  struct adam_var {
    Array weights, grads, m, v;
    adam_var(Array &_w, Array &_g) : weights(_w), grads(_g) {
      init();
    }
    cudaError_t init() {
      auto retval = cudaSuccess;
      GUARD_CU (m.Allocate (weights.GetSize (), util::DEVICE))
      GUARD_CU (m.ForEach ([]__host__ __device__(ValueT &x) { x = 0; }))
      GUARD_CU (v.Allocate (weights.GetSize (), util::DEVICE))
      GUARD_CU(v.ForEach ([]__host__ __device__(ValueT &x) { x = 0; }))
      return retval;
    }
  };

  struct sprmul : module {
    typedef app::sparseMatMul::Problem<SpmatT> ProblemT;
    typedef app::sparseMatMul::Enactor<ProblemT> EnactorT;

    SpmatT *a;
    util::Array1D<SizeT, ValueT> b, c, b_grad, c_grad;
    ProblemT *problem;
    EnactorT *enactor;
    int num_nodes, in_dim, out_dim;

    sprmul(Parameters &p, SpmatT &_a, util::Array1D<SizeT, ValueT> &_b, util::Array1D<SizeT, ValueT> &_b_grad,
        util::Array1D<SizeT, ValueT> &_c, util::Array1D<SizeT, ValueT> &_c_grad, int _in_dim, int _out_dim) :
    a(&_a), b(_b), c(_c), b_grad(_b_grad), c_grad(_c_grad), in_dim(_in_dim), out_dim(_out_dim) {
      num_nodes = _a.nodes;
      problem = new ProblemT(p);
      enactor = new EnactorT();

      problem->Init(_a, in_dim, out_dim);
      enactor->Init(*problem);
    }

    virtual void forward(bool train) override {
      problem->Reset(1, b, c);
      enactor->Reset();
      enactor->Enact();
    }

    virtual void backward() override {
      problem->Reset(0, c_grad, b_grad);
      enactor->Reset();
      enactor->Enact();
    }
  };

  struct graph_sum : module {
    typedef app::graphsum::Problem<GraphT> ProblemT;
    typedef app::graphsum::Enactor<ProblemT> EnactorT;

    GraphT *a;
    util::Array1D<SizeT, ValueT> b, c, b_grad, c_grad;
    ProblemT *problem;
    EnactorT *enactor;
    int dim;

    graph_sum(Parameters &p, GraphT &_a, util::Array1D<SizeT, ValueT> &_b, util::Array1D<SizeT, ValueT> &_b_grad,
           util::Array1D<SizeT, ValueT> &_c, util::Array1D<SizeT, ValueT> &_c_grad, int _dim) :
        a(&_a), b(_b), c(_c), b_grad(_b_grad), c_grad(_c_grad), dim(_dim) {
      problem = new ProblemT(p);
      enactor = new EnactorT();

      problem->Init(_a, dim);
      enactor->Init(*problem);
    }

    virtual void forward(bool train) override {
      problem->Reset(1, b, c);
      enactor->Reset();
      enactor->Enact();
    }

    virtual void backward() override {
      problem->Reset(0, c_grad, b_grad);
      enactor->Reset();
      enactor->Enact();
    }
  };

  struct cross_entropy : module {
    typedef app::CrossEntropyLoss::Problem<GraphT> ProblemT;
    typedef app::CrossEntropyLoss::Enactor<ProblemT> EnactorT;

    GraphT dummy;
    util::Array1D<SizeT, ValueT> logits, grad;
    util::Array1D<SizeT, int> truth;
    ProblemT *problem;
    EnactorT *enactor;
    int dim;

    cross_entropy(Parameters &p, Array _logits, Array _grad,
                  util::Array1D<SizeT, int> _truth, int num_nodes, int num_classes,
                  bool training=true) :
        logits(_logits), grad(_grad), truth(_truth) {
      problem = new ProblemT(p);
      enactor = new EnactorT();

      problem->Init(dummy, num_nodes, num_classes, logits, grad, truth, training);
      enactor->Init(*problem);
    }

    virtual void forward(bool train) override {
      problem->Reset(train);
      enactor->Reset();
      enactor->Enact();
    }

    virtual void backward() override {}

    virtual double GetLoss() override {
      double loss;
      problem->Extract (&loss);
      return loss;
    }
  };

  struct mat_mul : module {
    typedef app::graphsum::Problem<GraphT> ProblemT;
    typedef app::graphsum::Enactor<ProblemT> EnactorT;

    util::Array1D<SizeT, ValueT> b, c, b_grad, c_grad, a, a_grad;
    ProblemT *problem;
    EnactorT *enactor;
    int m, n, p;

    mat_mul(util::Array1D<SizeT, ValueT> &_a, util::Array1D<SizeT, ValueT> &_a_grad,
            util::Array1D<SizeT, ValueT> &_b, util::Array1D<SizeT, ValueT> &_b_grad,
            util::Array1D<SizeT, ValueT> &_c, util::Array1D<SizeT, ValueT> &_c_grad, int _m, int _n, int _p) :
        a(_a), b(_b), c(_c), a_grad(_a_grad), b_grad(_b_grad), c_grad(_c_grad), m(_m), n(_n), p(_p) {}

    virtual void forward(bool train) override {
      dofw();
    }

    virtual void backward() override {
      dobw();
    }

    cudaError_t dofw() {
      cudaError_t retval = cudaSuccess;
      auto &b = this->b, &c = this->c;
      auto &p = this->p, &n = this->n;

      GUARD_CU(
          c.ForEach([]__host__ __device__(ValueT &x) {
        x = 0;
      })
      )

      // Calculating matrix multiplication
      GUARD_CU(a.ForAll(
          [b, c, p, n]__host__ __device__(ValueT *a_, const SizeT pos) {
        int i = pos / n, j = pos % n;
        for (int k = 0; k < p; k++) {
          atomicAdd(c + i * p + k, a_[pos] * b[j * p + k]);
          //            printf("i: %d\n", i * p + k);
        }
      }, m * n, util::DEVICE));

      return retval;
    }

    cudaError_t dobw() {
      cudaError_t retval = cudaSuccess;

      GUARD_CU(
          a_grad.ForEach (
              []__host__ __device__(ValueT &x) { x = 0; }
      )
      )

      GUARD_CU(
          b_grad.ForEach (
              []__host__ __device__(ValueT &x) { x = 0; }
      )
      )

      auto &b_grad = this->b_grad, &c_grad = this->c_grad, &b = this->b, &a = this->a;
      auto &p = this->p, &n = this->n;
      // Calculating matrix multiplication
      GUARD_CU(a_grad.ForAll(
          [b_grad, c_grad, p, n, a, b]__host__ __device__(ValueT *a_, const SizeT pos) {
        int i = pos / n, j = pos % n;
        ValueT tmp = 0;
        for (int k = 0; k < p; k++) {
          tmp += c_grad[i * p + k] * b[j * p + k];
          atomicAdd(b_grad + j * p + k, a[pos] * c_grad[i * p + k]);
//          printf("c[%d], b[%d], a[%d]\n", i * p + k, j * p + k, pos);
        }
        a_[i * n + j] = tmp;
      }, m * n, util::DEVICE));
//      b_grad.Print();

      return retval;
    }
  };

  struct relu : module {
    util::Array1D<SizeT, ValueT> a, a_grad;
    util::Array1D<SizeT, bool> keep;
    int len;

    relu(util::Array1D<SizeT, ValueT> &_a, util::Array1D<SizeT, ValueT> &_a_grad, int _len) :
        a(_a), a_grad(_a_grad), len(_len) {
      keep.Allocate (a.GetSize (), util::DEVICE);
    };

    virtual void forward(bool train) override {
      dofw(train);
    }

    virtual void backward() override {
      dobw();
    }

    cudaError_t dofw(bool train) {
      cudaError_t retval = cudaSuccess;

      GUARD_CU(a.ForEach(keep,
          [train]__host__ __device__(ValueT &x, bool &k) {
          if (train) k = x > 0;
          if (!k) x = 0;
        }, a.GetSize (), util::DEVICE))
      ValueT res[a.GetSize ()];
      GUARD_CU (a.SetPointer (res, a.GetSize (), util::HOST))
      GUARD_CU (a.Move(util::DEVICE, util::HOST))
      std::ofstream out("relu_forward");
      out.precision(4);
      out << std::fixed;
      for (auto i = 0; i < a.GetSize (); i++) out << res[i] << '\n';
      a.UnSetPointer (util::HOST);

      return retval;
    }

    cudaError_t dobw() {
      cudaError_t retval = cudaSuccess;

      GUARD_CU(a_grad.ForEach(keep,
          []__host__ __device__(ValueT &grad, bool &k) {
        if (!k) grad = 0;
      }, a_grad.GetSize (), util::DEVICE))

      return retval;
    }
  };

  struct dropout : module {
    Array mask, data, *grad;
    ValueT p;
    curandGenerator_t gen;
    dropout(Array _data, Array *_grad, ValueT _p) : p(_p) {
      data = _data;
      grad = _grad;
      if (grad) mask.Allocate(data.GetSize (), util::DEVICE);
      curandCreateGenerator (&gen, CURAND_RNG_PSEUDO_DEFAULT);
    };
    virtual void forward(bool train) override {
      if (train) dofw();
    }
    virtual void backward() override {
      dobw();
    }
    cudaError_t dofw() {
      auto retval = cudaSuccess;
      curandGenerateUniformDouble(gen, mask.GetPointer(util::DEVICE), mask.GetSize ());
      ValueT scale = 1 / (1 - p);
      auto &p = this->p;
      GUARD_CU (mask.ForEach (data,
          [p, scale]__host__ __device__(ValueT &x, ValueT &d) {
            d *= x >= p ? scale : 0;
          }
          ))
      return retval;
    }

    cudaError_t dobw() {
      auto retval = cudaSuccess;
      if (!grad) return retval;
      ValueT scale = 1 / (1 - p);
      auto &p = this->p;
      GUARD_CU (mask.ForEach (*grad,
          [p, scale]__host__ __device__(ValueT &x, ValueT &g) {
        g *= x >= p ? scale : 0;
      }))
    }
  };

  /**
   * @brief Data structure containing graphsum-specific data on indivual GPU.
   */
  struct DataSlice : BaseDataSlice {
    typedef util::Array1D<SizeT, ValueT> Array;

    double learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 5e-4;

    std::vector<module*> modules;
    std::vector<adam_var> vars;
    util::Array1D<SizeT, int> truth, wrong, cnt, label, split;
    Array penalty, w0, xw0, Axw0, Axw0w1, AAxw0w1, w1;
    Array w0_grad, xw0_grad, Axw0_grad, Axw0w1_grad, AAxw0w1_grad, w1_grad, in_feature;
    SpmatT *x_ptr;

    int in_dim, hid_dim, out_dim, num_nodes, max_iter;
    bool training;
    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice() {
//      loss.SetName ("loss");
      split.SetName ("split");
      in_feature.SetName ("in_feature");
      label.SetName ("label");
      penalty.SetName ("penalty");
      wrong.SetName ("wrong");
      cnt.SetName ("cnt");
      truth.SetName ("truth");
      w0.SetName ("w0");
      w1.SetName ("w1");
      xw0.SetName ("xw0");
      Axw0.SetName ("Axw0");
      Axw0w1.SetName ("Axw0w1");
      AAxw0w1.SetName ("AAxw0w1");
      w0_grad.SetName ("w0_grad");
      w1_grad.SetName ("w1_grad");
      xw0_grad.SetName ("xw0_grad");
      Axw0_grad.SetName ("Axw0_grad");
      Axw0w1_grad.SetName ("Axw0w1_grad");
      AAxw0w1_grad.SetName ("AAxw0w1_grad");
      penalty.SetName("penalty");
    }

    /*
     * @brief Default destructor
     */
    virtual ~DataSlice() { Release(); }

    /*
     * @brief Releasing allocated memory space
     * @param[in] target      The location to release memory from
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Release(util::Location target = util::LOCATION_ALL) {
      cudaError_t retval = cudaSuccess;
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx));

      GUARD_CU(BaseDataSlice ::Release(target));
      return retval;
    }

    /**
     * @brief initializing graphsum-specific data on each gpu
     * @param     sub_graph   Sub graph on the GPU.
     * @param[in] num_gpus    Number of GPUs
     * @param[in] gpu_idx     GPU device index
     * @param[in] target      Targeting device location
     * @param[in] flag        Problem flag containling options
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Init(GraphT &sub_graph, util::Parameters &parameters, SpmatT &x, int *_truth,
                     int num_gpus = 1, int gpu_idx = 0,
                     util::Location target = util::DEVICE,
                     ProblemFlag flag = Problem_None) {
      cudaError_t retval = cudaSuccess;
      this->in_dim = parameters.Get<int>("in_dim");
      this->hid_dim = parameters.Get<int>("hid_dim");
      this->out_dim = parameters.Get<int>("out_dim");
      this->max_iter = parameters.Get<int>("max_iter");
      this->training = parameters.Get<bool>("training");
      this->learning_rate = parameters.Get<double>("learning_rate");
      this->num_nodes = sub_graph.nodes;
      this->x_ptr = &x;

      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));

      GUARD_CU (in_feature.Allocate (x.edges))
      GUARD_CU (penalty.Allocate (1))
      GUARD_CU (wrong.Allocate (1))
      GUARD_CU (cnt.Allocate (1))
      GUARD_CU (label.Allocate (num_nodes))
      GUARD_CU (truth.Allocate (num_nodes))
      GUARD_CU (split.Allocate (num_nodes))
      GUARD_CU (cnt.Allocate (1))
      GUARD_CU (w0.Allocate (in_dim * hid_dim))
      GUARD_CU (w1.Allocate (hid_dim * out_dim))
      GUARD_CU (xw0.Allocate (num_nodes * hid_dim))
      GUARD_CU (Axw0.Allocate (num_nodes * hid_dim))
      GUARD_CU (Axw0w1.Allocate (num_nodes * out_dim))
      GUARD_CU (AAxw0w1.Allocate (num_nodes * out_dim))
      GUARD_CU (w0_grad.Allocate (in_dim * hid_dim))
      GUARD_CU (w1_grad.Allocate (hid_dim * out_dim))
      GUARD_CU (xw0_grad.Allocate (num_nodes * hid_dim))
      GUARD_CU (Axw0_grad.Allocate (num_nodes * hid_dim))
      GUARD_CU (Axw0w1_grad.Allocate (num_nodes * out_dim))
      GUARD_CU (AAxw0w1_grad.Allocate (num_nodes * out_dim))

      GUARD_CU (label.SetPointer(_truth, num_nodes, util::HOST))
      GUARD_CU (label.Move(util::HOST, util::DEVICE))
//      GUARD_CU (truth.Print ())

      curandGenerator_t gen;
      curandCreateGenerator (&gen, CURAND_RNG_PSEUDO_DEFAULT);
      curandGenerateUniformDouble(gen, w0.GetPointer(util::DEVICE), w0.GetSize ());

      ValueT range = sqrt (6.0 / (in_dim + hid_dim));
      GUARD_CU (w0.ForEach (
          [range]__host__ __device__(ValueT &x) {
            x = (x - 0.5) * range * 2;
          }
      ))
      curandGenerateUniformDouble(gen, w1.GetPointer(util::DEVICE), w1.GetSetted ());

      range = sqrt (6.0 / (out_dim + hid_dim));
      GUARD_CU (w1.ForEach (
          [range]__host__ __device__(ValueT &x) {
            x = (x - 0.5) * range * 2;
          }
      ))

      Array *dummy = nullptr;
      modules.push_back(new dropout(x.edge_values, dummy, 0.5));
      modules.push_back(new sprmul(parameters, x, w0, w0_grad, xw0, xw0_grad, in_dim, hid_dim));
      modules.push_back(new graph_sum(parameters, sub_graph, xw0, xw0_grad, Axw0, Axw0_grad, hid_dim));
      modules.push_back(new relu(Axw0, Axw0_grad, num_nodes * hid_dim));
      modules.push_back(new dropout(Axw0, &Axw0_grad, 0.5));
      modules.push_back(new mat_mul(Axw0, Axw0_grad, w1, w1_grad, Axw0w1, Axw0w1_grad, num_nodes, hid_dim, out_dim));
      modules.push_back(new graph_sum(parameters, sub_graph, Axw0w1, Axw0w1_grad, AAxw0w1, AAxw0w1_grad, out_dim));
      modules.push_back(new cross_entropy(parameters, AAxw0w1, AAxw0w1_grad, truth, num_nodes, out_dim));

      GUARD_CU(x.edge_values.ForEach(in_feature,
          []__host__ __device__(ValueT &src, ValueT &dst) {
        dst = src;
      }, x.edge_values.GetSize(), util::DEVICE))

      vars.emplace_back(w0, w0_grad);
      vars.emplace_back(w1, w1_grad);

      GUARD_CU(sub_graph.Move(util::HOST, target, this->stream));
      return retval;
    }  // Init

    /**
     * @brief Reset problem function. Must be called prior to each run.
     * @param[in] target      Targeting device location
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Reset(util::Location target = util::DEVICE) {
      cudaError_t retval = cudaSuccess;
      return retval;
    }
  };  // DataSlice

  // Members
  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;
  SpmatT in;
  int *_truth;

  // Methods

  /**
   * @brief graphsum default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {}

  /**
   * @brief graphsum default destructor
   */
  virtual ~Problem() { Release(); }

  /*
   * @brief Releasing allocated memory space
   * @param[in] target      The location to release memory from
   * \return    cudaError_t Error message(s), if any
   */
  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    if (data_slices == NULL) return retval;
    for (int i = 0; i < this->num_gpus; i++)
      GUARD_CU(data_slices[i].Release(target));

    if ((target & util::HOST) != 0 &&
        data_slices[0].GetPointer(util::DEVICE) == NULL) {
      delete[] data_slices;
      data_slices = NULL;
    }
    GUARD_CU(BaseProblem::Release(target));
    return retval;
  }

  /**
   * \addtogroup PublicInterface
   * @{
   */


  cudaError_t Extract(ValueT *_w0, ValueT *_w1,
                      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    SizeT nodes = this->org_graph->nodes;

    if (this->num_gpus == 1) {
      auto &data_slice = data_slices[0][0];

      // Set device
      if (target == util::DEVICE) {
        GUARD_CU(util::SetDevice(this->gpu_idx[0]));
        GUARD_CU (data_slice.w0.SetPointer(_w0, data_slice.in_dim * data_slice.hid_dim))
        GUARD_CU (data_slice.w0.Move(util::DEVICE, util::HOST))
        GUARD_CU (data_slice.w1.SetPointer(_w1, data_slice.out_dim * data_slice.hid_dim))
        GUARD_CU (data_slice.w1.Move(util::DEVICE, util::HOST))
      }
    }

    return retval;
  }

  cudaError_t read_feature(Parameters &p, SpmatT &g, int *&truth) {
    typedef typename SpmatT::CsrT CsrT;

    auto retval = cudaSuccess;

    int n_rows = 0, nnz = 0, dim;
    static std::vector<typename GraphT::SizeT> indptr, indices;
    static std::vector<int> labels;
    static std::vector<ValueT> feature_val;
    indptr.push_back(0);
    std::ifstream svmlight_file(p.Get<std::string>("feature_file"));

    int max_idx = 0, max_label = 0;
    while(true) {
        std::string line;
        getline(svmlight_file, line);
        if (svmlight_file.eof()) break;
        indptr.push_back(indptr.back());
        std::istringstream ss(line);

        int label = -1;
        ss >> label;
        labels.push_back(label);
        if (ss.fail()) continue;
        max_label = std::max(max_label, label);

        while (true) {
            std::string kv;
            ss >> kv;
            if(ss.fail()) break;
            std::istringstream kv_ss(kv);

            int k;
            float v;
            char col;
            kv_ss >> k >> col >> v;

            feature_val.push_back(v);
            indices.push_back(k);
            indptr.back() += 1;
            max_idx = std::max(max_idx, k);
          }
      }
    n_rows = indptr.size() - 1;
    nnz = indices.size();
    dim = max_idx + 1;

    p.Set("in_dim", dim);
    p.Set("out_dim", max_label + 1);

    g.CsrT::Allocate(n_rows, nnz, gunrock::util::HOST);
    g.CsrT::row_offsets.SetPointer(indptr.data(), n_rows + 1, gunrock::util::HOST);
    g.CsrT::column_indices.SetPointer(indices.data(), nnz, gunrock::util::HOST);
    g.CsrT::edge_values.SetPointer(feature_val.data(), nnz, gunrock::util::HOST);
    g.nodes = n_rows;
//    std::cout << "n_rows: " << n_rows << '\n';
    g.edges = nnz;
    gunrock::graphio::LoadGraph(p, g);
//    g.Move(util::HOST, util::DEVICE);

//    for (auto e : labels) std::cout << e << '\n'; std::cout << '\n';
    truth = labels.data();

    return retval;
  }

  void get_split(Parameters &p, std::vector<int> &split) {
    std::ifstream split_file(p.Get<std::string>("split_file"));

    while (true) {
      std::string line;
      getline(split_file, line);
      if (split_file.eof()) break;
      int cur_split = stoi(line);
      split.push_back(cur_split);
    }
  }

  /**
   * @brief      initialization function.
   *
   * @param      graph   The graph that SSSP processes on
   * @param[in]  dim     The dimension of the feature vector
   * @param[in]  target  The target
   * @param[in]  Location  Memory location to work on
   *
   * @return     cudaError_t Error message(s), if any
   */
  cudaError_t Init(GraphT &graph, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseProblem::Init(graph, target));
    data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

      auto &data_slice = data_slices[gpu][0];

      read_feature (BaseProblem::parameters, in, _truth);
//      for (int i = 0; i < graph.nodes; i++) std::cout << _truth[i] << '\n';
//      std::cout << '\n';
      std::vector<int> split;
      get_split(BaseProblem::parameters, split);
      data_slice.Init(this->sub_graphs[gpu], BaseProblem::parameters, in, _truth, this->num_gpus,
                               this->gpu_idx[gpu], target, this->flag);
      GUARD_CU (data_slice.split.SetPointer(split.data (), graph.nodes, util::HOST))
      GUARD_CU (data_slice.split.Move(util::HOST, util::DEVICE))
    }  // end for (gpu)

    return retval;
  }

  /**
   * @brief Reset problem function. Must be called prior to each run.
   * @param[in] location Memory location to work on
   * \return cudaError_t Error message(s), if any
   */
  cudaError_t Reset(util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      // Set device
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU(data_slices[gpu]->Reset(target));
      GUARD_CU(data_slices[gpu].Move(util::HOST, target));
    }

    if (target & util::DEVICE) {
      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    }

    return retval;
  }

  /** @} */
};

}  // namespace graphsum
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
