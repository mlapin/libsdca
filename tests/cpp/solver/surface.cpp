// The public C++ surface: shared-name objectives, the row loss API, seeds and logging.
#include <sstream>
#include <typeinfo>

#include "sdca/loss.h"
#include "sdca/solver.h"
#include "test_util.h"

namespace {

// Captured before any test runs: logging is off until a program enables it.
const sdca::logging::level initial_level = sdca::logging::get_level();

// Records the objective type a visitor receives.
struct type_of {
  std::string& name;
  template <typename Objective>
  void operator()(const Objective&) const {
    name = typeid(Objective).name();
  }
};

template <typename Objective>
std::string name_of() {
  return typeid(Objective).name();
}

template <typename Data, typename Result>
std::string visited(const sdca::objective_spec& spec, const bool multilabel) {
  std::string name;
  if (multilabel)
    sdca::visit_multilabel_objective<Data, Result>(spec, type_of{name});
  else
    sdca::visit_multiclass_objective<Data, Result>(spec, type_of{name});
  return name;
}

// Trains spec on a small problem; returns the duals.
struct train {
  const std::vector<double>& x;
  const std::vector<sdca::size_type>& y;
  sdca::size_type d, m;
  std::uint64_t seed;
  std::vector<double>& a;
  sdca::solver_status& status;
  template <typename Objective>
  void operator()(Objective objective) const {
    const sdca::size_type n = y.size();
    std::vector<double> w(d * m);
    a.assign(n * m, 0);
    auto ctx = sdca::make_context(sdca::make_input_feature(n, d, x.data()),
                                  sdca::make_output_multiclass(y.begin(), y.end(), m),
                                  std::move(objective), w.data(), a.data());
    ctx.criteria.seed = seed;
    ctx.criteria.max_epoch = 3;
    ctx.criteria.eval_epoch = 0;
    ctx.criteria.tol = 0;
    sdca::make_solver(ctx).solve();
    status = ctx.status;
  }
};

// Checks the parameters of the visited objectives.
struct check {
  template <typename Objective>
  void operator()(const Objective&) const {
    ADD_FAILURE() << "unexpected objective type";
  }
  void operator()(const sdca::l2_topk_hinge_alpha_smooth<double, double>& o) const {
    EXPECT_EQ(3, o.c);
    EXPECT_EQ(1u, o.k);
    EXPECT_EQ(.5, o.gamma);
  }
  void operator()(const sdca::l2_topk_hinge_beta<double, double>& o) const {
    EXPECT_EQ(2, o.c);
    EXPECT_EQ(4u, o.k);
  }
};

}  // namespace

TEST(SurfaceTest, logging_is_off_by_default_and_redirects) {
  EXPECT_EQ(sdca::logging::level::none, initial_level);
  const sdca::logging::level level = sdca::logging::get_level();
  std::ostringstream out;
  sdca::logging::set_stream(out);
  sdca::logging::set_level(sdca::logging::level::none);
  SDCA_LOG_WARNING << "hidden";
  EXPECT_TRUE(out.str().empty());
  sdca::logging::set_level(sdca::logging::level::warning);
  SDCA_LOG_WARNING << "shown";
  SDCA_LOG_INFO << "hidden";
  EXPECT_EQ("shown", out.str());
  sdca::logging::set_stream(std::cout);
  sdca::logging::set_level(level);
}

TEST(SurfaceTest, shared_name_objectives_visit_their_types) {
  typedef double D;
  EXPECT_EQ((name_of<sdca::l2_cross_entropy<D, D>>()),
            (visited<D, D>(sdca::cross_entropy_objective(), false)));
  EXPECT_EQ((name_of<sdca::l2_topk_cross_entropy<D, D>>()),
            (visited<D, D>(sdca::topk_cross_entropy_objective(1, 2), false)));
  EXPECT_EQ((name_of<sdca::l2_topk_hinge_alpha<D, D>>()),
            (visited<D, D>(sdca::multiclass_hinge_objective(), false)));
  EXPECT_EQ((name_of<sdca::l2_topk_hinge_alpha_smooth<D, D>>()),
            (visited<D, D>(sdca::multiclass_hinge_objective(1, .5), false)));
  EXPECT_EQ((name_of<sdca::l2_topk_hinge_alpha<float, double>>()),
            (visited<float, double>(sdca::topk_hinge_alpha_objective(1, 2), false)));
  EXPECT_EQ((name_of<sdca::l2_topk_hinge_alpha_smooth<D, D>>()),
            (visited<D, D>(sdca::topk_hinge_alpha_objective(1, 2, 1), false)));
  EXPECT_EQ((name_of<sdca::l2_topk_hinge_beta<D, D>>()),
            (visited<D, D>(sdca::topk_hinge_beta_objective(1, 2), false)));
  EXPECT_EQ((name_of<sdca::l2_topk_hinge_beta_smooth<D, D>>()),
            (visited<D, D>(sdca::topk_hinge_beta_objective(1, 2, 1), false)));
  EXPECT_EQ((name_of<sdca::l2_multilabel_hinge<D, D>>()),
            (visited<D, D>(sdca::multilabel_hinge_objective(), true)));
  EXPECT_EQ((name_of<sdca::l2_multilabel_hinge_smooth<D, D>>()),
            (visited<D, D>(sdca::multilabel_hinge_objective(1, .1), true)));
  EXPECT_EQ((name_of<sdca::l2_multilabel_cross_entropy<D, D>>()),
            (visited<D, D>(sdca::multilabel_cross_entropy_objective(2), true)));

  // multiclass_hinge has k = 1; the visited objective carries C, k and gamma.
  sdca::visit_multiclass_objective<D, D>(sdca::multiclass_hinge_objective(3, .5), check());
  sdca::visit_multiclass_objective<D, D>(sdca::topk_hinge_beta_objective(2, 4), check());

  // The other output's kinds, and invalid arguments.
  std::string name;
  EXPECT_THROW(
      (sdca::visit_multiclass_objective<D, D>(sdca::multilabel_hinge_objective(), type_of{name})),
      std::invalid_argument);
  EXPECT_THROW(
      (sdca::visit_multilabel_objective<D, D>(sdca::cross_entropy_objective(), type_of{name})),
      std::invalid_argument);
  sdca::objective_spec prox = sdca::cross_entropy_objective();
  prox.kind = sdca::loss_kind::proximal_cross_entropy;
  EXPECT_THROW((sdca::visit_multiclass_objective<D, D>(prox, type_of{name})),
               std::invalid_argument);
  EXPECT_THROW(sdca::cross_entropy_objective(0), std::invalid_argument);
  EXPECT_THROW(sdca::topk_cross_entropy_objective(1, 0), std::invalid_argument);
  EXPECT_THROW(sdca::multiclass_hinge_objective(1, -1), std::invalid_argument);
  EXPECT_THROW(sdca::topk_hinge_beta_objective(1, 2, std::numeric_limits<double>::infinity()),
               std::invalid_argument);
}

TEST(SurfaceTest, seeds_set_the_order_of_the_examples) {
  const sdca::size_type n = 60, d = 4, m = 3;
  std::mt19937 gen(3);
  std::normal_distribution<double> noise(0, 1);
  std::vector<double> x(n * d);
  std::vector<sdca::size_type> y(n);
  for (sdca::size_type i = 0; i < n; ++i) {
    y[i] = i % m;
    for (sdca::size_type j = 0; j < d; ++j) x[d * i + j] = noise(gen) + (j == y[i] ? 1 : 0);
  }
  std::vector<double> a0, a1, b0, b1;
  sdca::solver_status status;
  const sdca::logging::level level = sdca::logging::get_level();
  sdca::logging::set_level(sdca::logging::level::none);
  const sdca::objective_spec spec = sdca::topk_hinge_alpha_objective(1, 1);
  sdca::visit_multiclass_objective<double, double>(spec, train{x, y, d, m, 0, a0, status});
  EXPECT_EQ(sdca::solver_status::max_epoch, status);
  sdca::visit_multiclass_objective<double, double>(spec, train{x, y, d, m, 0, a1, status});
  sdca::visit_multiclass_objective<double, double>(spec, train{x, y, d, m, 7, b0, status});
  sdca::visit_multiclass_objective<double, double>(spec, train{x, y, d, m, 7, b1, status});
  EXPECT_EQ(a0, a1);
  EXPECT_EQ(b0, b1);
  EXPECT_NE(a0, b0);
  // minstd_rand's states: seeds equal modulo 2^31 - 2 give the same order.
  std::vector<double> c;
  const std::uint64_t period = 2147483646u;
  sdca::visit_multiclass_objective<double, double>(spec, train{x, y, d, m, 7 + period, c, status});
  EXPECT_EQ(b0, c);
  sdca::logging::set_level(level);
}

TEST(SurfaceTest, row_losses_take_the_target_in_class_order) {
  const std::size_t m = 6;
  const double z[] = {.3, -1.2, 2.5, .7, 2.4, -.1};
  sdca::loss_parameters top2;
  top2.k = 2;
  sdca::loss_parameters smooth;
  smooth.gamma = .5;
  sdca::loss_parameters smooth_top2 = top2;
  smooth_top2.gamma = .25;
  sdca::loss_parameters proximal;
  proximal.alpha = .7;
  const struct {
    sdca::loss_kind kind;
    sdca::loss_parameters params;
  } cases[] = {{sdca::loss_kind::cross_entropy, sdca::loss_parameters()},
               {sdca::loss_kind::topk_cross_entropy, top2},
               {sdca::loss_kind::multiclass_hinge, sdca::loss_parameters()},
               {sdca::loss_kind::multiclass_hinge, smooth},
               {sdca::loss_kind::topk_hinge_alpha, top2},
               {sdca::loss_kind::topk_hinge_beta, smooth_top2},
               {sdca::loss_kind::proximal_cross_entropy, proximal}};
  for (const auto& c : cases) {
    for (std::size_t label = 0; label < m; ++label) {
      std::vector<double> g(m), permuted(m), order(m), expected(m);
      const double value = sdca::loss_and_gradient(c.kind, m, z, label, g.data(), c.params);
      // The permuted kernel: the label first, then the others in order.
      std::vector<std::size_t> index(1, label);
      for (std::size_t j = 0; j < m; ++j)
        if (j != label) index.push_back(j);
      for (std::size_t j = 0; j < m; ++j) order[j] = z[index[j]];
      const double reference = sdca::detail::permuted_loss_and_gradient(
          c.kind, m, 1, order.data(), permuted.data(), c.params.k ? c.params.k : 1, c.params.gamma,
          c.params.alpha);
      for (std::size_t j = 0; j < m; ++j) expected[index[j]] = permuted[j];
      EXPECT_EQ(reference, value) << sdca::loss_name(c.kind) << " label " << label;
      EXPECT_EQ(expected, g) << sdca::loss_name(c.kind) << " label " << label;
      // Values alone.
      EXPECT_EQ(value, sdca::loss_and_gradient(c.kind, m, z, label, static_cast<double*>(nullptr),
                                               c.params));
    }
  }

  // Label sets: the labels first, then the others, each in order.
  const std::size_t labels[] = {1, 4};
  for (const auto kind :
       {sdca::loss_kind::multilabel_hinge, sdca::loss_kind::multilabel_cross_entropy}) {
    std::vector<double> g(m), permuted(m), order(m), expected(m);
    sdca::loss_parameters params;
    if (kind == sdca::loss_kind::multilabel_hinge) params.gamma = .3;
    const double value = sdca::loss_and_gradient(kind, m, z, labels, 2, g.data(), params);
    const std::size_t index[] = {1, 4, 0, 2, 3, 5};
    for (std::size_t j = 0; j < m; ++j) order[j] = z[index[j]];
    const double reference = sdca::detail::permuted_loss_and_gradient(
        kind, m, 2, order.data(), permuted.data(), 1, params.gamma);
    for (std::size_t j = 0; j < m; ++j) expected[index[j]] = permuted[j];
    EXPECT_EQ(reference, value) << sdca::loss_name(kind);
    EXPECT_EQ(expected, g) << sdca::loss_name(kind);
  }
  // multilabel_cross_entropy takes every class: its target is uniform.
  const std::size_t all[] = {0, 1, 2, 3, 4, 5};
  std::vector<double> g(m);
  EXPECT_NO_THROW(
      sdca::loss_and_gradient(sdca::loss_kind::multilabel_cross_entropy, m, z, all, m, g.data()));
  EXPECT_THROW(sdca::loss_and_gradient(sdca::loss_kind::multilabel_hinge, m, z, all, m, g.data()),
               std::invalid_argument);
}

TEST(SurfaceTest, row_losses_reject_invalid_arguments) {
  const std::size_t m = 4;
  const double z[] = {1, 2, 3, 4};
  double g[4];
  sdca::loss_parameters k2;
  k2.k = 2;
  sdca::loss_parameters gamma;
  gamma.gamma = 1;
  sdca::loss_parameters alpha;
  alpha.alpha = 2;
  sdca::loss_parameters extended;
  extended.projection = sdca::projection_precision::extended;
  const std::size_t sets[] = {2, 1}, twice[] = {1, 1}, outside[] = {0, 4};
  // Labels out of range; k missing, out of range or not taken.
  EXPECT_THROW(sdca::loss_and_gradient(sdca::loss_kind::cross_entropy, m, z, 4, g),
               std::invalid_argument);
  EXPECT_THROW(sdca::loss_and_gradient(sdca::loss_kind::topk_hinge_beta, m, z, 0, g),
               std::invalid_argument);
  sdca::loss_parameters k4;
  k4.k = 4;
  EXPECT_THROW(sdca::loss_and_gradient(sdca::loss_kind::topk_hinge_beta, m, z, 0, g, k4),
               std::invalid_argument);
  EXPECT_THROW(sdca::loss_and_gradient(sdca::loss_kind::cross_entropy, m, z, 0, g, k2),
               std::invalid_argument);
  // Parameters of other losses.
  EXPECT_THROW(sdca::loss_and_gradient(sdca::loss_kind::cross_entropy, m, z, 0, g, gamma),
               std::invalid_argument);
  EXPECT_THROW(sdca::loss_and_gradient(sdca::loss_kind::multiclass_hinge, m, z, 0, g, alpha),
               std::invalid_argument);
  EXPECT_THROW(sdca::loss_and_gradient(sdca::loss_kind::multiclass_hinge, m, z, 0, g, extended),
               std::invalid_argument);
  EXPECT_NO_THROW(
      sdca::loss_and_gradient(sdca::loss_kind::multilabel_hinge, m, z, sets + 1, 1, g, extended));
  // Targets of the other kind, invalid sets, and the distillation losses.
  EXPECT_THROW(sdca::loss_and_gradient(sdca::loss_kind::multilabel_hinge, m, z, 0, g),
               std::invalid_argument);
  EXPECT_THROW(sdca::loss_and_gradient(sdca::loss_kind::cross_entropy, m, z, sets + 1, 1, g),
               std::invalid_argument);
  EXPECT_THROW(sdca::loss_and_gradient(sdca::loss_kind::multilabel_cross_entropy, m, z, sets, 2, g),
               std::invalid_argument);
  EXPECT_THROW(
      sdca::loss_and_gradient(sdca::loss_kind::multilabel_cross_entropy, m, z, twice, 2, g),
      std::invalid_argument);
  EXPECT_THROW(
      sdca::loss_and_gradient(sdca::loss_kind::multilabel_cross_entropy, m, z, outside, 2, g),
      std::invalid_argument);
  EXPECT_THROW(sdca::loss_and_gradient(sdca::loss_kind::multilabel_cross_entropy, m, z, sets, 0, g),
               std::invalid_argument);
  EXPECT_THROW(sdca::loss_and_gradient(sdca::loss_kind::distillation, m, z, 0, g),
               std::invalid_argument);
  EXPECT_THROW(sdca::loss_and_gradient(sdca::loss_kind::cross_entropy, 1, z, 0, g),
               std::invalid_argument);
}
