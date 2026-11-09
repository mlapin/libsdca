#ifndef SDCA_TESTS_REGRESSION_SOLVER_H
#define SDCA_TESTS_REGRESSION_SOLVER_H

#ifndef __FAST_MATH__
TEST(NumericalRegression, all_capped_offset_survives_cancelling_scores) {
  for (double huge : {1e16, 1e30, 1e100}) {
    std::vector<double> competitors = {-huge, 1., huge};
    do {
      double scores[] = {-1., competitors[0], competitors[1], competitors[2]};
      double reference[] = {-1., 0., 1., 0.}, grad[4], expected[4];
      const auto kind = sdca::loss_kind::topk_cross_entropy;
      const double loss = sdca::loss_and_gradient(kind, 4, 1, scores, grad, 3);
      const double ref = sdca::loss_and_gradient(kind, 4, 1, reference, expected, 3);
      EXPECT_NEAR(loss, ref, 2e-14);
      for (int j = 0; j < 4; ++j) EXPECT_NEAR(grad[j], expected[j], 2e-14);
      double a[4] = {0}, b[4] = {0};
      sdca::l2_entropy_topk<double, double> objective(10., 3);
      objective.update_dual_variables(4, 1., a, scores);
      objective.update_dual_variables(4, 1., b, reference);
      for (int j = 0; j < 4; ++j) EXPECT_NEAR(a[j], b[j], 2e-13);
      auto projected = competitors;
      auto th =
          sdca::thresholds_topk_entropy_offset<double>(projected.begin(), projected.end(), 3, 1.);
      sdca::prox(th, projected.begin(), projected.end());
      for (double value : projected) EXPECT_NEAR(value, expected[1], 2e-14);
    } while (std::next_permutation(competitors.begin(), competitors.end()));
  }
}
#endif

#ifndef __FAST_MATH__
TEST(NumericalRegression, all_capped_warm_start_keeps_score_and_dual_means) {
  for (double huge : {1e16, 1e30, 1e100}) {
    double scores[] = {0., huge, 1., -huge}, reference[] = {0., 0., 1., 0.};
    double a[] = {3., -1., -1., -1.}, b[] = {3., -1., -1., -1.};
    // Both blocks have mean 1/3 + 4. Combining each score with its dual
    // before averaging discards the small correction beside +/-huge.
    sdca::l2_entropy_topk<double, double> objective(10., 3);
    objective.update_dual_variables(4, 1., a, scores);
    objective.update_dual_variables(4, 1., b, reference);
    for (int j = 0; j < 4; ++j) EXPECT_NEAR(a[j], b[j], 2e-13);
  }
}
#endif

#ifndef __FAST_MATH__
TEST(NumericalRegression, all_capped_direct_update_validates_inputs) {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (double bad : {inf, -inf, nan})
    for (int j = 0; j < 4; ++j) {
      double scores[4] = {0}, a[4] = {0};
      scores[j] = bad;
      sdca::l2_entropy_topk<double, double> objective(1., 3);
      EXPECT_THROW(objective.update_dual_variables(4, 1., a, scores), std::invalid_argument);
      scores[j] = 0.;
      a[j] = bad;
      EXPECT_THROW(objective.update_dual_variables(4, 1., a, scores), std::invalid_argument);
    }
  for (double c : {0., -1., inf, nan, 1e308}) {
    double scores[4] = {0}, a[4] = {0};
    using Objective = sdca::l2_entropy_topk<double, double>;
    EXPECT_THROW(
        {
          Objective objective(c, 3);
          objective.update_dual_variables(4, 2., a, scores);
        },
        std::invalid_argument);
  }
}
#endif

TEST(NumericalRegression, all_capped_finite_scale_with_overflowing_coefficient) {
  for (int k : {2, 3})
    for (double mean : {-1., -1e-20, 0., 1e-20, 1.}) {
      const double c = 1.5e308;
      std::vector<double> scores(std::size_t(k + 1), mean), a(std::size_t(k + 1), 0.);
      scores[0] = 0.;
      const sdca::l2_entropy_topk<double, double> objective(c, std::size_t(k));
      if (!wide_long_double()) {
        // c (1 + 1/k) overflows double, the range of the retry.
        EXPECT_THROW(objective.update_dual_variables(k + 1, 1., a.data(), scores.data()),
                     std::overflow_error);
        continue;
      }
      objective.update_dual_variables(k + 1, 1., a.data(), scores.data());
      EXPECT_GT(a[0], 400.);
      EXPECT_LT(a[0], 600.);
      // Independent scalar KKT equation in the returned dual mass.
      const long double mass = static_cast<long double>(a[0]) / static_cast<long double>(c);
      const long double residual =
          std::log(mass) - std::log1p(-mass) + (1 + 1.L / k) * static_cast<long double>(a[0]) -
          std::log(static_cast<long double>(k)) - static_cast<long double>(mean);
      EXPECT_NEAR(static_cast<double>(residual), 0., 1e-10);
      for (int j = 1; j <= k; ++j) EXPECT_NEAR(a[std::size_t(j)], -a[0] / k, 1e-12);
    }
}

TEST(NumericalRegression, all_capped_near_unit_mass) {
  for (int k : {2, 3, 10})
    for (double c : {1e14, 1e16, 1e18, 1e30}) {
      const double mean = c * (1. + 1. / k);
      std::vector<double> scores(std::size_t(k + 1), mean), a(std::size_t(k + 1), 0.);
      scores[0] = 0.;
      sdca::l2_entropy_topk<double, double>(c, std::size_t(k))
          .update_dual_variables(k + 1, 1., a.data(), scores.data());
      // At this boundary 1-s is at most log(coefficient)/coefficient.
      // The old shifted solve loses the log-odds and can miss s by 1e-8.
      EXPECT_NEAR(a[0] / c, 1., 1e-11);
      std::vector<double> v(std::size_t(k), mean), out(std::size_t(k), 0.);
      sdca::topk_entropy_block(std::size_t(k), v.data(), std::size_t(k), c, out.data());
      for (int j = 0; j < k; ++j) EXPECT_NEAR(out[std::size_t(j)] * k, 1., 1e-11);
    }
}

#ifndef __FAST_MATH__
TEST(NumericalRegression, all_capped_nonuniform_warm_start) {
  // Nonuniform caller-supplied blocks must still take the compensated path.
  for (int changed : {1, 2, 3}) {
    double scores[] = {0., 1e30, 1., -1e30}, reference[] = {0., 0., 1., 0.};
    double a[] = {3., -1., -1., -1.}, b[] = {3., -1., -1., -1.};
    a[changed] += .5;
    a[changed == 3 ? 1 : changed + 1] -= .5;
    sdca::l2_entropy_topk<double, double> objective(10., 3);
    objective.update_dual_variables(4, 1., a, scores);
    objective.update_dual_variables(4, 1., b, reference);
    for (int j = 0; j < 4; ++j) EXPECT_NEAR(a[j], b[j], 2e-13);
  }
}
#endif

#ifndef __FAST_MATH__
TEST(NumericalRegression, entropy_block_extremes_and_validation) {
  // Confidently wrong: the label probability underflows, the block is still
  // the softmax complement; confidently right: the residual is kept.
  for (double gap : {40., 100., 700.}) {
    double z[] = {-gap, 0, 0}, a[] = {0, 0, 0};
    sdca::l2_entropy<double, double>(1).update_dual_variables(3, 1., a, z);
    EXPECT_NEAR(a[0], 1., 1e-15);
    EXPECT_NEAR(a[1], -.5, 1e-15);
    double w[] = {gap, 0, 0}, b[] = {0, 0, 0};
    sdca::l2_entropy<double, double>(1).update_dual_variables(3, 1., b, w);
    EXPECT_GT(b[0], 0);
    EXPECT_NEAR(b[0] / (2 * std::exp(-gap)), 1., 1e-6);
  }
  for (double alpha : {1e-300, 1e-30, 1e30, 1e300}) {
    std::vector<double> v = {0, 1, -1}, probabilities;
    double a[3], t;
    sdca::entropy_prox_dual(3, 1, 1., alpha, v, a, &probabilities, &t);
    EXPECT_NEAR(probabilities[0] + probabilities[1] + probabilities[2], 1., 1e-14);
    // One entry is reconstructed from the others: zero-sum to one rounding.
    EXPECT_LE(std::abs(a[0] + a[1] + a[2]),
              std::numeric_limits<double>::epsilon() * (std::abs(a[1]) + std::abs(a[2])));
    EXPECT_NEAR(probabilities[1], -a[1], 1e-15);
    EXPECT_TRUE(std::isfinite(t));
  }
  std::vector<double> bad = {0, std::numeric_limits<double>::quiet_NaN()};
  double a[2];
  EXPECT_THROW(sdca::entropy_prox_dual(2, 1, 1., 1., bad, a), std::invalid_argument);
  std::vector<double> ok = {0, 1};
  EXPECT_THROW(sdca::entropy_prox_dual(2, 2, 1., 1., ok, a), std::invalid_argument);
  EXPECT_THROW(sdca::entropy_prox_dual(2, 1, 1., 0., ok, a), std::invalid_argument);
  EXPECT_THROW(sdca::entropy_prox_dual(2, 1, 1., 1., std::vector<double>{0}, a),
               std::invalid_argument);
}
#endif

// Labels far below the other scores (confidently wrong examples, or float
// scores in the thousands): the blocks are taken relative to the largest
// score, so the duals stay feasible and the other classes keep the split
// that their own score differences give.
TEST(NumericalRegression, far_off_label_scores) {
  const double eps = std::numeric_limits<double>::epsilon();
  for (double c : {1e-3, 1., 100.})
    for (double gap : {700., 1e16, 1e300}) {
      SCOPED_TRACE(::testing::Message() << "C=" << c << " gap=" << gap);
      std::vector<double> z = {-gap, 0, 1}, old(3, 0.), a(3, 0.);
      sdca::l2_entropy<double, double> objective(c);
      objective.update_dual_variables(3, 1., a.data(), z.data());
      EXPECT_LE(a[0], c * (1 + 4 * eps));
      EXPECT_NEAR(a[0], c, 4 * eps * c);
      EXPECT_TRUE(std::isfinite(objective.dual_loss(3, a.data())));
      check_entropy_block_kkt(3, 1, c, z, old, a, 1e-13);
      std::vector<double> zm = {-gap, -gap, 0, 1}, oldm(4, 0.), am(4, 0.);
      sdca::l2_multilabel_entropy<double, double>(c).update_dual_variables(4, 2, 1., am.data(),
                                                                           zm.data());
      EXPECT_LE(am[0], c / 2 * (1 + 4 * eps));
      EXPECT_LE(am[1], c / 2 * (1 + 4 * eps));
      check_entropy_block_kkt(4, 2, c, zm, oldm, am, 1e-13);
      for (std::size_t k : {1, 2, 3}) {
        std::vector<double> zt = {-gap, 0, 1, 2, -1}, at(5, 0.), g(5);
        sdca::l2_entropy_topk<double, double>(c, k).update_dual_variables(5, 1., at.data(),
                                                                          zt.data());
        EXPECT_LE(at[0], c * (1 + 4 * eps));
        for (std::size_t j = 0; j < 5; ++j) zt[j] += at[j];
        sdca::loss_and_gradient(sdca::loss_kind::topk_cross_entropy, 5, 1, zt.data(), g.data(), k);
        for (std::size_t j = 0; j < 5; ++j)
          EXPECT_NEAR(at[j], -c * g[j], 1e-13 * c) << "k=" << k << " j=" << j;
      }
    }
  // In float, scores in the thousands resolve only to about 1e-3: the
  // split between the other classes still follows their exact differences.
  std::vector<float> z = {-1e4f, 0.f, 1e-4f}, old(3, 0.f), a(3, 0.f);
  sdca::l2_entropy<float, float>(1).update_dual_variables(3, 1.f, a.data(), z.data());
  EXPECT_LE(a[0], 1 + 4 * std::numeric_limits<float>::epsilon());
  EXPECT_NEAR(double(a[2]) / double(a[1]), std::exp(1e-4 + (double(a[2]) - double(a[1]))), 2e-6);
  check_entropy_block_kkt(3, 1, 1., z, old, a, 2e-6);
  // The top-k entropy loss gradient at such scores: capped softmax of the
  // other classes, in their order.
  double s[] = {-1e16, .88, 1.73, 2.44, -1.05}, g[5];
  sdca::loss_and_gradient(sdca::loss_kind::topk_cross_entropy, 5, 1, s, g, 1);
  long double den(0);
  for (double value : s) den += std::exp((long double)value - 2.44L);
  for (std::size_t j = 0; j < 5; ++j)
    EXPECT_NEAR(g[j], double(std::exp((long double)s[j] - 2.44L) / den) - (j == 0), 1e-15);
  sdca::loss_and_gradient(sdca::loss_kind::topk_cross_entropy, 5, 1, s, g, 2);
  EXPECT_NEAR(g[3], .5, 1e-15);
  EXPECT_NEAR(g[1] / g[2], std::exp(.88 - 1.73), 1e-15);
  EXPECT_NEAR(g[1] + g[2] + g[3] + g[4], 1., 1e-15);
}

// Saturated entries far above the free ones: the threshold's origin must
// come from the free entries, or they lose their values.
TEST(NumericalRegression, knapsack_origin_near_free_entries) {
  std::vector<double> x = {1e16, 1e16, 1e16, .5, .3};
  sdca::prox_knapsack_eq(x.begin(), x.end(), 0., .3, 1.);
  EXPECT_NEAR(x[3], .1, 1e-15);
  EXPECT_EQ(x[4], 0.);
  EXPECT_NEAR(std::accumulate(x.begin(), x.end(), 0.), 1., 1e-15);
  std::vector<float> y = {1e7f, 1e7f, 1e7f, .5f, .3f};
  sdca::prox_knapsack_eq(y.begin(), y.end(), 0.f, .3f, 1.f);
  EXPECT_NEAR(y[3], .1f, 1e-6f);
  EXPECT_EQ(y[4], 0.f);
}

// Direct-objective probes across high margins, common offsets, and large C.
TEST(NumericalRegression, high_margin_offset_and_large_c_probes) {
  sdca::l2_entropy<double, double> obj(1);
  double z_zero[] = {0, -40, -40}, z_shift[] = {20, -20, -20};
  double ref = std::log1p(2 * std::exp(-40.));
  EXPECT_NEAR(obj.primal_loss(3, z_zero) / ref, 1., 1e-13);
  EXPECT_NEAR(obj.primal_loss(3, z_shift) / ref, 1., 1e-13);
  double z_common[] = {1e16, 1e16, 1e16};
  EXPECT_NEAR(obj.primal_loss(3, z_common), std::log(3.), 1e-15);

  // Symmetric optimum a = 2C exp(-3a/2) / (1 + 2 exp(-3a/2)) at C = 1e18.
  sdca::l2_entropy<double, double> large(1e18);
  double a[] = {0, 0, 0}, z[] = {0, 0, 0};
  large.update_dual_variables(3, 1., a, z);
  EXPECT_NEAR(a[0], 25.923031359473278, 1e-12);
  EXPECT_EQ(a[0], -a[1] - a[2]);
  double feasible[] = {26, -13, -13};
  EXPECT_NEAR(large.dual_loss(3, feasible), 1036.9211402272135, 1e-9);

  sdca::l2_multilabel_entropy<float, double> multi(1);
  float na[] = {0, 0, 0}, nz[] = {0, 1e-10f, -40};
  multi.update_dual_variables(3, 2, 1.f, na, nz);
  EXPECT_NEAR(double(na[0]), 1.66666677e-11, 1e-16);
  EXPECT_NEAR(double(na[1]), -1.66666656e-11, 1e-16);

  for (int m = 1; m <= 30; ++m) {
    float v[] = {-float(m)};
    sdca::prox_topk_entropy_biased<float>(v, v + 1, 1, 1.f);
    EXPECT_GT(v[0], 0.f);
    double s = double(v[0]);
    EXPECT_NEAR(2 * s + std::log(s) - std::log1p(-s) + m, 0., 1e-5);
  }
}

TEST(NumericalRegression, topk_caps_far_above_the_free_scores) {
  // Scores 1e15 apart: the top k share the mass; the threshold must be
  // resolved relative to the largest free score, 1e16 below the maximum.
  {
    double z[] = {-1e16, 0, -1e15, -2e15, -3e15, -4e15, -5e15, -6e15}, a[8] = {0};
    sdca::l2_entropy_topk<double, double>(1e-3, 3).update_dual_variables(8, 1., a, z);
    EXPECT_NEAR(a[0], 1e-3, 1e-17);
    for (int j = 1; j <= 3; ++j) EXPECT_NEAR(a[j], -1e-3 / 3, 1e-17);
    for (int j = 4; j < 8; ++j) EXPECT_EQ(a[j], 0.);
  }
  {
    double z[] = {0, 1e15, 2e15, 3e15, 4e15, -1e15}, a[6] = {0};
    sdca::l2_entropy_topk<double, double>(1e-3, 4).update_dual_variables(6, 1., a, z);
    EXPECT_NEAR(a[0], 1e-3, 1e-17);
    for (int j = 1; j <= 4; ++j) EXPECT_NEAR(a[j], -2.5e-4, 1e-17);
    EXPECT_EQ(a[5], 0.);
  }
}

TEST(NumericalRegression, topk_large_alpha_balancing_large_offset) {
  // C = 1e18 against a label 1e16 below: the mass is unsaturated (8.75e-3)
  // while its log-odds relative to the largest score is -1e16. The duals
  // differ by the score differences, as the quadratic term dictates.
  const double z[] = {-1e16, 0, 1, 0, 2, 0, 0, 1};
  for (int mode : {0, 1, 2}) {
    std::vector<double> a(8, 0.);
    if (mode == 2) {
      std::vector<double> zz(z, z + 8);
      sdca::l2_entropy_topk<double, double>(1e18, 5).update_dual_variables(8, 1., a.data(),
                                                                           zz.data());
      EXPECT_NEAR(a[0] / 8.750000000000006e15, 1., 1e-13);
      for (int j = 1; j < 8; ++j) {
        EXPECT_NEAR(a[j] / -1.25e15, 1., 1e-13);
        EXPECT_NEAR(a[j] - a[1], -(z[j] - z[1]), .5);
      }
    } else {
      std::vector<float> zz(z, z + 8), af(8, 0.f);
      if (mode == 0)
        sdca::l2_entropy_topk<float, float>(1e18f, 5).update_dual_variables(8, 1.f, af.data(),
                                                                            zz.data());
      else
        sdca::l2_entropy_topk<float, double>(1e18, 5).update_dual_variables(8, 1.f, af.data(),
                                                                            zz.data());
      EXPECT_NEAR(double(af[0]) / 8.75e15, 1., 1e-5);
      for (int j = 1; j < 8; ++j) EXPECT_NEAR(double(af[j]) / -1.25e15, 1., 1e-5);
    }
  }
  // The uniform branch (k = n) in the same regime.
  double y[] = {-1e16, 0, 1}, b[3] = {0};
  sdca::l2_entropy_topk<double, double>(1e18, 2).update_dual_variables(3, 1., b, y);
  EXPECT_NEAR(b[0] / 6666666666666671., 1., 1e-13);
  EXPECT_NEAR(b[1] / -3333333333333335.5, 1., 1e-13);
  EXPECT_EQ(b[1], b[2]);
}

// The default-scratch wrappers accept any random-access iterator, including
// raw pointers and the split two-block forms, and agree with vector input.
#define SDCA_EXPECT_ITERATOR_AGREEMENT(call)                      \
  {                                                               \
    std::vector<double> v(x);                                     \
    std::deque<double> d(x.begin(), x.end());                     \
    double* p = raw.data();                                       \
    std::copy(x.begin(), x.end(), p);                             \
    {                                                             \
      auto first = v.begin(), middle = first + 2, last = v.end(); \
      (void)middle;                                               \
      call;                                                       \
    }                                                             \
    {                                                             \
      auto first = d.begin(), middle = first + 2, last = d.end(); \
      (void)middle;                                               \
      call;                                                       \
    }                                                             \
    {                                                             \
      double *first = p, *middle = p + 2, *last = p + x.size();   \
      (void)middle;                                               \
      call;                                                       \
    }                                                             \
    for (std::size_t j = 0; j < x.size(); ++j) {                  \
      EXPECT_NEAR(v[j], d[j], ieee ? 0. : 1e-12) << #call;        \
      EXPECT_NEAR(v[j], p[j], ieee ? 0. : 1e-12) << #call;        \
    }                                                             \
  }
TEST(NumericalRegression, prox_wrappers_accept_any_random_access_iterator) {
  const std::vector<double> x = {.3, -1.2, 2.5, .7, -.4, 1.1};
  std::vector<double> raw(x.size());
  // -Ofast may vectorize the iterator types differently.
#ifdef __FAST_MATH__
  const bool ieee = false;
#else
  const bool ieee = true;
#endif
  SDCA_EXPECT_ITERATOR_AGREEMENT(sdca::prox_entropy(first, last));
  SDCA_EXPECT_ITERATOR_AGREEMENT(sdca::prox_entropy_norm(first, last));
  SDCA_EXPECT_ITERATOR_AGREEMENT(sdca::prox_knapsack_eq(first, last, 0., 1., 1.));
  SDCA_EXPECT_ITERATOR_AGREEMENT(sdca::prox_knapsack_le(first, last, 0., 1., 1.));
  SDCA_EXPECT_ITERATOR_AGREEMENT(sdca::prox_knapsack_le_biased(first, last, 0., 1., 1., 1.));
  SDCA_EXPECT_ITERATOR_AGREEMENT(sdca::prox_topk_cone(first, last, 2));
  SDCA_EXPECT_ITERATOR_AGREEMENT(sdca::prox_topk_cone_biased(first, last, 2, 1.));
  SDCA_EXPECT_ITERATOR_AGREEMENT(sdca::prox_topk_simplex(first, last, 2));
  SDCA_EXPECT_ITERATOR_AGREEMENT(sdca::prox_topk_simplex_biased(first, last, 2, 1., 1.));
  SDCA_EXPECT_ITERATOR_AGREEMENT(sdca::prox_topk_entropy(first, last, 2));
  SDCA_EXPECT_ITERATOR_AGREEMENT(sdca::prox_topk_entropy_biased(first, last, 2, 1.));
  SDCA_EXPECT_ITERATOR_AGREEMENT(sdca::prox_two_simplex(first, middle, last));
  SDCA_EXPECT_ITERATOR_AGREEMENT(sdca::prox_two_simplex(first, middle, middle, last));
  SDCA_EXPECT_ITERATOR_AGREEMENT(sdca::prox_two_simplex_sort(first, middle, last));
  SDCA_EXPECT_ITERATOR_AGREEMENT(sdca::prox_two_entropy(first, middle, last, 2.));
}
#undef SDCA_EXPECT_ITERATOR_AGREEMENT

// One float-storage block update with Result chosen by the caller, against
// the double-storage solution: the largest error and largest capped entry.
template <typename Result>
static void float_storage_error(int n, std::size_t k, double c, const std::vector<float>& z,
                                double& block, double& error, double& cap, double& a0) {
  std::vector<float> zf(z), a(static_cast<std::size_t>(n), 0.f);
  std::vector<double> zd(z.begin(), z.end()), ad(static_cast<std::size_t>(n), 0.);
  if (k == 1) {
    sdca::l2_entropy<float, Result>(Result(c)).update_dual_variables(n, 1.f, a.data(), zf.data());
    sdca::l2_entropy<double, double>(c).update_dual_variables(n, 1., ad.data(), zd.data());
  } else {
    sdca::l2_entropy_topk<float, Result>(Result(c), k)
        .update_dual_variables(n, 1.f, a.data(), zf.data());
    sdca::l2_entropy_topk<double, double>(c, k).update_dual_variables(n, 1., ad.data(), zd.data());
  }
  block = error = cap = 0;
  for (int j = 0; j < n; ++j) {
    block = std::max(block, std::abs(ad[std::size_t(j)]));
    error = std::max(error, std::abs(double(a[std::size_t(j)]) - ad[std::size_t(j)]));
    if (j > 0) cap = std::max(cap, -double(a[std::size_t(j)]));
  }
  a0 = double(a[0]);
}

// Float storage with a double Result solves its blocks in double: within a
// few float ulps of the block, with feasible caps. A float Result solves in
// float, accurate to the rounding of the scores and the solver's mass
// tolerance (64 n eps).
TEST(NumericalRegression, float_storage_blocks_match_double_storage) {
  std::mt19937 rng(7);
  std::normal_distribution<double> normal(0, 1);
  const double ulp = std::pow(2., -24);
  for (int it = 0; it < 600; ++it) {
    const int n = it % 3 == 0 ? 3 : it % 3 == 1 ? 10 : 100;
    const double scale = std::pow(10., it % 4), c = std::pow(10., double(it % 5) - 1);
    const std::size_t k = it % 2 ? 1 : std::size_t(2 + it % std::max(1, n - 3));
    std::vector<float> z(static_cast<std::size_t>(n));
    double largest = 0;
    for (auto& v : z) {
      v = float(normal(rng) * scale);
      largest = std::max(largest, std::abs(double(v)));
    }
    double block, error, cap, a0;
    float_storage_error<double>(n, k, c, z, block, error, cap, a0);
    if (block < 1e-30) continue;
    const std::string where = "n=" + std::to_string(n) + " k=" + std::to_string(k) +
                              " C=" + std::to_string(c) + " scale=" + std::to_string(scale);
    EXPECT_LE(error, 8 * ulp * block) << "double Result " << where;
    if (k > 1) {
      EXPECT_LE(cap, a0 / double(k) * (1 + 8 * ulp)) << "double Result " << where;
    }
    float_storage_error<float>(n, k, c, z, block, error, cap, a0);
    EXPECT_LE(error, (64 * n + 16 * largest) * ulp * block) << "float Result " << where;
    if (k > 1) {
      EXPECT_LE(cap, a0 / double(k) * (1 + 64 * n * ulp)) << "float Result " << where;
    }
  }
}

#ifndef __FAST_MATH__
// Capped-softmax fixed point of a top-k entropy update at its updated
// scores (norm2 = 1), in long double: the error of the block relative to
// its largest entry. Well conditioned while the updated scores are moderate.
static double topk_fixed_point_error(const std::vector<double>& z, const std::vector<double>& a,
                                     double c, std::size_t k) {
  using L = long double;
  const std::size_t n = z.size() - 1;
  std::vector<L> v(n);
  for (std::size_t j = 0; j < n; ++j) v[j] = (L(z[j + 1]) + L(a[j + 1])) - (L(z[0]) + L(a[0]));
  std::vector<std::size_t> order(n);
  for (std::size_t j = 0; j < n; ++j) order[j] = j;
  std::sort(order.begin(), order.end(), [&](std::size_t i, std::size_t j) { return v[i] > v[j]; });
  std::size_t u = 0;
  long double lse = 0;
  for (; u < k; ++u) {
    long double top = v[order[u]], sum = 0;
    for (std::size_t j = u; j < n; ++j) sum += std::exp(v[order[j]] - top);
    lse = top + std::log(sum);
    if (v[order[u]] <= lse - std::log(L(k - u))) break;
  }
  long double capped = 0;
  for (std::size_t j = 0; j < u; ++j) capped += v[order[j]];
  const L K = L(k);
  const long double odds = (capped + (K - u) * (lse - std::log(K - u))) / K + std::log(K);
  const long double mass = 1 / (1 + std::exp(-odds));
  std::vector<long double> expected(n + 1);
  expected[0] = L(c) * mass;
  for (std::size_t j = 0; j < n; ++j) {
    const std::size_t i = order[j];
    expected[i + 1] = -L(c) * (j < u ? mass / K : mass * (1 - L(u) / K) * std::exp(v[i] - lse));
  }
  long double scale = 0, error = 0;
  for (std::size_t j = 0; j <= n; ++j) {
    scale = std::max(scale, std::abs(expected[j]));
    error = std::max(error, std::abs(expected[j] - L(a[j])));
  }
  return double(error / scale);
}

// Moderate scores at a huge C put the mass near 1/C. The equal-share step
// must reach it (a bisection in the mass did not, from C = 1e150 on), and
// steps below the resolution of the log-odds must end the iteration.
TEST(NumericalRegression, topk_huge_c_with_moderate_scores) {
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> score(-100, 100);
  for (double c : {1e100, 1e150, 1e200, 1e300}) {
    for (int it = 0; it < 120; ++it) {
      const std::size_t n = 4 + std::size_t(it) % 37, k = 2 + std::size_t(it) % (n - 3);
      std::vector<double> z(n), a(n, 0.);
      for (auto& x : z) x = score(rng);
      std::vector<double> scores(z);
      try {
        sdca::l2_entropy_topk<double, double>(c, k).update_dual_variables(n, 1., a.data(),
                                                                          scores.data());
      } catch (const std::runtime_error&) {
        // With the range of double the retry may fail loudly above C = 1e20.
        EXPECT_FALSE(wide_long_double()) << "C=" << c << " n=" << n << " k=" << k;
        continue;
      }
      EXPECT_LE(topk_fixed_point_error(z, a, c, k), 1e-9) << "C=" << c << " n=" << n << " k=" << k;
    }
  }
}

// The range every platform supports, also where long double is double.
TEST(NumericalRegression, topk_large_c_within_the_range_of_double) {
  std::mt19937 rng(6);
  std::uniform_real_distribution<double> score(-100, 100);
  for (double c : {1e10, 1e15, 1e20}) {
    for (int it = 0; it < 120; ++it) {
      const std::size_t n = 4 + std::size_t(it) % 37, k = 2 + std::size_t(it) % (n - 3);
      std::vector<double> z(n), a(n, 0.);
      for (auto& x : z) x = score(rng);
      std::vector<double> scores(z);
      sdca::l2_entropy_topk<double, double>(c, k).update_dual_variables(n, 1., a.data(),
                                                                        scores.data());
      EXPECT_LE(topk_fixed_point_error(z, a, c, k), 1e-9) << "C=" << c << " n=" << n << " k=" << k;
    }
  }
}
#endif

// Rows of zero norm get the block that maximizes the dual on its own: the
// gap closes, for features and for a Gram matrix with a zero row.
template <typename Context>
sdca::solver_status solve_tightly(Context ctx) {
  ctx.criteria.epsilon = 1e-12;
  ctx.criteria.eval_epoch = 1;
  sdca::make_solver(ctx).solve();
  return ctx.status;
}

template <typename Output, typename Objective>
void expect_zero_row_solved(const Objective& obj, const Output& out) {
  const std::size_t m = 4, n = 4, d = 2;
  std::vector<double> x = {0, 0, 1, 0.5, -0.5, 1, 0.3, -1}, gram(n * n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      gram[i * n + j] = x[i * d] * x[j * d] + x[i * d + 1] * x[j * d + 1];
  std::vector<double> a(n * m), w(d * m);
  EXPECT_EQ(sdca::solver_status::solved,
            solve_tightly(sdca::make_context(sdca::make_input_feature(d, n, x.data()), Output(out),
                                             Objective(obj), a.data(), w.data())))
      << obj.to_string();
  EXPECT_NEAR(0, std::accumulate(a.begin(), a.begin() + m, 0.0), 1e-15);
  EXPECT_GT(std::abs(a[0]) + std::abs(a[1]), 0);
  std::fill(a.begin(), a.end(), 0);
  EXPECT_EQ(sdca::solver_status::solved,
            solve_tightly(sdca::make_context(sdca::make_input_kernel(n, gram.data()), Output(out),
                                             Objective(obj), a.data())))
      << obj.to_string() << " (kernel)";
  EXPECT_GT(std::abs(a[0]) + std::abs(a[1]), 0);
}

TEST(NumericalRegression, zero_norm_rows_reach_the_optimum) {
  std::vector<sdca::size_type> labels = {1, 0, 2, 3}, sets = {0, 2, 1, 3, 0, 1, 2},
                               offsets = {0, 2, 4, 5, 7};
  const sdca::multiclass_output y(4, labels);
  const sdca::multilabel_output multi(4, sets, offsets);
  expect_zero_row_solved(sdca::l2_entropy<double, double>(1), y);
  expect_zero_row_solved(sdca::l2_entropy_topk<double, double>(1, 2), y);
  expect_zero_row_solved(sdca::l2_hinge_topk<double, double>(1, 2), y);
  expect_zero_row_solved(sdca::l2_topk_hinge<double, double>(1, 2), y);
  for (double gamma : {0.5, 10.0}) {
    expect_zero_row_solved(sdca::l2_hinge_topk_smooth<double, double>(1, 2, gamma), y);
    expect_zero_row_solved(sdca::l2_topk_hinge_smooth<double, double>(1, 2, gamma), y);
    expect_zero_row_solved(sdca::l2_multilabel_hinge_smooth<double, double>(1, gamma), multi);
  }
  expect_zero_row_solved(sdca::l2_multilabel_entropy<double, double>(1), multi);
  expect_zero_row_solved(sdca::l2_multilabel_hinge<double, double>(1), multi);

  // The zero row of entropy: c (e_y - 1/m), with its label in place.
  double x[] = {0, 1}, w[2] = {}, a[4] = {};
  std::vector<sdca::size_type> y2 = {1, 0};
  auto ctx = sdca::make_context(sdca::make_input_feature(1, 2, x), sdca::multiclass_output(2, y2),
                                sdca::l2_entropy<double, double>(1), a, w);
  ctx.criteria.epsilon = 1e-12;
  ctx.criteria.eval_epoch = 1;
  sdca::make_solver(ctx).solve();
  EXPECT_EQ(sdca::solver_status::solved, ctx.status);
  EXPECT_EQ(-0.5, a[0]);
  EXPECT_EQ(0.5, a[1]);
}

TEST(NumericalRegression, failed_step_ends_the_solve_as_failed) {
  double x[] = {1, 2}, a[] = {0, 0, 0}, w[6] = {};
  std::vector<sdca::size_type> labels = {2};
  auto ctx =
      sdca::make_context(sdca::make_input_feature(2, 1, x), sdca::multiclass_output(3, labels),
                         failing_entropy<double, double>(), a, w);
  ctx.criteria.eval_epoch = 1;
  sdca::make_solver(ctx).solve();
  EXPECT_EQ(sdca::solver_status::failed, ctx.status);
  EXPECT_EQ(0u, ctx.epoch);
  ASSERT_FALSE(ctx.train.evals.empty());
  EXPECT_EQ(std::vector<double>(3, 0), std::vector<double>(a, a + 3));
}

TEST(NumericalRegression, zero_row_block_beyond_float_storage_is_rejected) {
  // c (e_y - 1/m) = [5e38, -5e38] overflows float: C must be normal in the
  // data type before the solve, and the block must not change.
  for (bool kernel : {false, true}) {
    float x[] = {0}, w[2] = {}, a[2] = {};
    std::vector<sdca::size_type> labels = {0};
    const sdca::multiclass_output y(2, labels);
    typedef sdca::l2_entropy<float, double> Objective;
    if (kernel) {
      auto ctx = sdca::make_context(sdca::make_input_kernel(1, x), sdca::multiclass_output(y),
                                    Objective(1e39), a);
      EXPECT_THROW(sdca::make_solver(ctx).solve(), std::invalid_argument);
    } else {
      auto ctx = sdca::make_context(sdca::make_input_feature(1, 1, x), sdca::multiclass_output(y),
                                    Objective(1e39), a, w);
      EXPECT_THROW(sdca::make_solver(ctx).solve(), std::invalid_argument);
    }
    EXPECT_EQ(0, a[0]);
    EXPECT_EQ(0, a[1]);
  }
}

TEST(NumericalRegression, entropy_labels_share_the_recovered_other_mass) {
  // Both label residuals round to zero, the other class keeps c p.
  const double v[] = {400, 400, -400};
  double out[3];
  sdca::entropy_prox_dual_ptr(3, 2, 1e200, 1.0, v, out);
  EXPECT_LT(out[2], 0);
  EXPECT_GT(-out[2], 1e-150);
  EXPECT_EQ(out[0], out[1]);
  EXPECT_NEAR(0, (out[0] + out[1] + out[2]) / out[2], 1e-15);
}

#ifndef __FAST_MATH__
template <typename T>
static void proximal_entropy_finite_label_mean() {
  const T limit = std::numeric_limits<T>::max(), eps = std::numeric_limits<T>::epsilon();
  const T z[] = {T(0), -limit, -limit, T(0)};
  T gradient[4];
  const double value = sdca::loss_and_gradient<T, T>(sdca::loss_kind::proximal_cross_entropy, 4, 3,
                                                     z, gradient, 1, 1, 1);
  // The two low-label probabilities round to zero. The finite mean is
  // 2*limit/3 plus bounded terms, despite an overflowing sum of label losses.
  const T expected = (limit / T(3)) * T(2);
  ASSERT_TRUE(std::isfinite(value));
  EXPECT_NEAR(value / double(expected), 1., 4 * double(eps));
  // Independently solve log(p0/(1-p0)) = 4/3 - 2*p0 at 100 digits.
  const double p0 = 0.5554032662653987;
  const double reference[] = {p0 - 1. / 3., -1. / 3., -1. / 3., 1. - p0};
  for (std::size_t j = 0; j < 4; ++j)
    EXPECT_NEAR(double(gradient[j]), reference[j], 8 * double(eps));
}

TEST(NumericalRegression, proximal_entropy_keeps_representable_label_means) {
  proximal_entropy_finite_label_mean<float>();
  proximal_entropy_finite_label_mean<double>();
}
#endif

TEST(NumericalRegression, proximal_entropy_value_at_extreme_scales) {
  // Large alpha: equal label scores reach log 2 (u = v + alpha a cancels).
  double z[] = {1e16, 0, 0}, g[3];
  EXPECT_NEAR(
      std::log(2.0),
      sdca::loss_and_gradient(sdca::loss_kind::proximal_cross_entropy, 3, 2, z, g, 1, 1, 1e50),
      1e-15);
  // a_j^2 underflows; the value is d^2 / (4 alpha) + log1p(exp(-d)) with
  // d + log d + log1p(exp(-d)) = log(2 alpha).
  double zero[] = {0, 0}, h[2];
  EXPECT_NEAR(
      1.1762794618348152e-295,
      sdca::loss_and_gradient(sdca::loss_kind::proximal_cross_entropy, 2, 1, zero, h, 1, 1, 1e300),
      1e-308);
  // Moderate scales agree with the cross-entropy at the updated scores.
  std::mt19937 gen(3);
  std::normal_distribution<double> normal(0, 3);
  for (double alpha : {1e-6, 0.3, 1.0, 40.0}) {
    for (std::size_t p : {1, 2}) {
      std::vector<double> s(6), grad(6), u(6);
      for (auto& v : s) v = normal(gen);
      const double value = sdca::loss_and_gradient(sdca::loss_kind::proximal_cross_entropy, 6, p,
                                                   s.data(), grad.data(), 1, 1, alpha);
      double penalty = 0;
      for (std::size_t j = 0; j < 6; ++j) {
        u[j] = s[j] - alpha * grad[j];
        penalty += alpha * grad[j] * grad[j] / 2;
      }
      EXPECT_NEAR(sdca::entropy_loss<double>(std::size_t(6), p, u.data()) + penalty, value,
                  1e-14 * value);
    }
  }
}

TEST(NumericalRegression, entropy_residual_of_a_subnormal_root) {
  // r = t / (alpha + 1/q) is subnormal: the bracket must not replace it by a midpoint.
  const double denorm = std::numeric_limits<double>::denorm_min();
  for (double alpha : {1e-6, 1.0, 1e4, 1e300}) {
    for (double t : {1e-312, 5e-320, 1e-300, 1e-290}) {
      const double r = sdca::detail::entropy_residual(t, alpha, 0.5);
      EXPECT_NEAR(t / (alpha + 2), r, 2 * denorm + 1e-15 * r) << alpha << " " << t;
    }
  }
  const float r = sdca::detail::entropy_residual(1e-40f, 1.0f, 0.5f);
  EXPECT_NEAR(1e-40f / 3, r, 2 * std::numeric_limits<float>::denorm_min());
}

// The proximal entropy of p labels at the largest score z[0] and other classes
// whose probabilities exp(z_j - z[0]) / p underflow (or nearly).
template <typename T>
static void check_underflowing_others(const std::vector<T>& z, std::size_t p, double alpha) {
  const std::size_t n = z.size();
  std::vector<T> g(n);
  T value = T(0);
  ASSERT_NO_THROW(value = T(sdca::loss_and_gradient<T, T>(sdca::loss_kind::proximal_cross_entropy,
                                                          n, p, z.data(), g.data(), 1, 1, alpha)))
      << alpha << " " << z[0];
  const double denorm = double(std::numeric_limits<T>::denorm_min());
  const double eps = double(std::numeric_limits<T>::epsilon());
  double sum = 0, mass = 0;
  for (std::size_t j = 0; j < n; ++j) {
    sum += double(g[j]);
    if (j < p) {
      EXPECT_LE(g[j], T(0)) << alpha << " " << j;
      continue;
    }
    const double v = double(z[j]) - double(z[0]), expected = std::exp(v) / double(p);
    mass += expected;
    // The rounding of the scaled score v - t, and of a subnormal result.
    EXPECT_NEAR(expected, double(g[j]), 2 * denorm + 4 * eps * (1 - v) * expected)
        << alpha << " " << z[0] << " " << j;
  }
  EXPECT_LE(std::abs(sum), 2 * denorm * double(n) + 4 * eps * mass) << alpha;
  // -log((1 - mass) / p) to first order in the other mass.
  const double expected = std::log(double(p)) + mass;
  EXPECT_NEAR(expected, double(value), 8 * eps * expected + 8 * denorm) << alpha << " " << z[0];
}

TEST(NumericalRegression, proximal_entropy_with_underflowing_other_classes) {
  // These did not converge (the inner residual of a subnormal root took a
  // bracket midpoint) or, at alpha = 1e300, returned a nonfinite loss.
  for (double alpha : {1e-6, 1e-2, 1.0, 30.0, 1e4, 1e300}) {
    check_underflowing_others<double>({720, 720, 0, 1, 2}, 2, alpha);
    check_underflowing_others<double>({745, 745, 0, 1, 2}, 2, alpha);
    check_underflowing_others<double>({745, 0, 1, 2, -1}, 1, alpha);
    // Here alpha times the other mass would shift the label score.
    if (alpha < 1e100) check_underflowing_others<double>({709, 0, 1, 2, -1}, 1, alpha);
  }
  for (double alpha : {1e-6, 1e-2, 1.0, 30.0, 1e4}) {
    check_underflowing_others<float>({100, 100, 0, 1, 2}, 2, alpha);
    check_underflowing_others<float>({100, 0, 1, 2, -1}, 1, alpha);
    check_underflowing_others<float>({90, 90, 0, 1, 2}, 2, alpha);
  }
}

#ifndef __FAST_MATH__
// A warm 4-class step of entropy, top-k (k=2 caps inactive, k=3 all capped)
// or multilabel (p=2) entropy at norm2 = 1.
template <typename Data, typename Result>
static std::vector<Data> warm_entropy_step(int objective, double c, std::vector<Data> a,
                                           std::vector<Data> z) {
  const Result cr = Result(c);
  if (objective == 0)
    sdca::l2_entropy<Data, Result>(cr).update_dual_variables(4, Data(1), a.data(), z.data());
  else if (objective == 3)
    sdca::l2_multilabel_entropy<Data, Result>(cr).update_dual_variables(4, 2, Data(1), a.data(),
                                                                        z.data());
  else
    sdca::l2_entropy_topk<Data, Result>(cr, std::size_t(objective + 1))
        .update_dual_variables(4, Data(1), a.data(), z.data());
  return a;
}

// Largest error relative to the largest reference entry, in units of eps.
template <typename Data, typename Reference>
static double block_eps_error(const std::vector<Data>& a, const std::vector<Reference>& ref) {
  double error = 0, block = 0;
  for (std::size_t j = 0; j < ref.size(); ++j) {
    error = std::max(error, std::abs(double(a[j]) - double(ref[j])));
    block = std::max(block, std::abs(double(ref[j])));
  }
  return error / block / double(std::numeric_limits<Data>::epsilon());
}

template <typename Data, typename Result>
static void warm_entropy_cancellation(const char* precision) {
  const double A = std::ldexp(1., 80);
  const Data s80 = Data(A), h80 = Data(A / 2);
  // Residual z - old = [0,0,0,2] at C = A: independent 100-digit KKT solve of
  // log(d_j) - log(C-S) + d_j + S = v_j - v_0 (d_j = -a_j, S = sum d).
  const std::vector<double> oracle = {40.14498811749243021, -12.76043681668961128,
                                      -12.76043681668961128, -14.62411448411320765};
  for (int objective : {0, 1}) {
    const auto a = warm_entropy_step<Data, Result>(objective, A, {s80, -h80, -h80, Data(0)},
                                                   {s80, -h80, -h80, Data(2)});
    EXPECT_LE(block_eps_error(a, oracle), 32) << precision << " objective " << objective;
  }
  // The same cancellation across scales equals the cold step on the residual,
  // to the resolution of the root solve: the warm step starts it from the old
  // block's threshold, the cold one from the softmax solution, and each stops
  // within |f| <= 8 eps of the residual mass (16 eps of the largest entry).
  const std::vector<Data> cold(4, Data(0)), residual = {Data(0), Data(0), Data(0), Data(2)};
  for (int power = 0; power <= 100; power += 4) {
    const Data s = Data(std::ldexp(1., power)), h = s / 2;
    for (int objective : {0, 1, 2, 3}) {
      const bool multi = objective == 3;
      const std::vector<Data> old =
          multi ? std::vector<Data>{h, h, -s, Data(0)} : std::vector<Data>{s, -h, -h, Data(0)};
      std::vector<Data> z(old);
      z[3] = Data(2);
      const auto warm = warm_entropy_step<Data, Result>(objective, double(s), old, z);
      const auto reference = warm_entropy_step<Data, Result>(objective, double(s), cold, residual);
      EXPECT_LE(block_eps_error(warm, reference), 32)
          << precision << " objective " << objective << " scale 2^" << power;
    }
  }
}

template <typename Data, typename Result>
static void warm_entropy_common_offset(const char* precision) {
  // Small duals beside a huge common score: forming z - old first loses them.
  // Oracle: independent 100-digit KKT solve at the residual -old, C = 1.
  const std::vector<double> oracle = {0.7744739448815888755, -0.2754728123133952801,
                                      -0.2495005662840967977, -0.2495005662840967977};
  // The same for the multilabel block (p = 2, uniform on the labels).
  const std::vector<double> multilabel = {0.2754097341474755993, 0.2754097341474755993,
                                          -0.3023343716165979108, -0.2484850966783532879};
  for (int power = 0; power <= 120; power += 4) {
    const Data offset = Data(std::ldexp(1., power));
    for (int objective : {0, 1, 2, 3}) {
      const std::vector<Data> old =
          objective == 3 ? std::vector<Data>{Data(.375), Data(.375), Data(-.5), Data(-.25)}
                         : std::vector<Data>{Data(.875), Data(-.375), Data(-.25), Data(-.25)};
      const std::vector<Data> residual = {-old[0], -old[1], -old[2], -old[3]};
      const auto warm =
          warm_entropy_step<Data, Result>(objective, 1, old, std::vector<Data>(4, offset));
      const auto cold =
          warm_entropy_step<Data, Result>(objective, 1, std::vector<Data>(4, Data(0)), residual);
      const std::string where = std::string(precision) + " objective " + std::to_string(objective) +
                                " offset 2^" + std::to_string(power);
      if (objective < 3) {
        EXPECT_LE(block_eps_error(warm, cold), 4) << where;
      }
      if (objective < 2) {
        EXPECT_LE(block_eps_error(warm, oracle), 8) << where;
      }
      if (objective == 3) {
        // In double the cold step stops 5 eps from the root, the warm one under 1.
        EXPECT_LE(block_eps_error(warm, multilabel), 8) << where;
        EXPECT_LE(block_eps_error(cold, multilabel), 8) << where;
      }
    }
  }
}

TEST(NumericalRegression, warm_entropy_keeps_residual_of_cancelling_scores) {
  warm_entropy_cancellation<double, double>("float64");
  warm_entropy_cancellation<float, double>("float32 storage / float64 compute");
  warm_entropy_cancellation<float, float>("float32");
}

// An evaluation of the entropy block solver expanded about an exact one, or
// started from it, against the exact evaluation at the same threshold, per class: the two
// differ by the rounding of the class's argument (eps |x| times dp/dx).
template <typename T>
static void expanded_entropy_evaluations(const char* precision) {
  typedef sdca::detail::entropy_block_solver<T, int> block_solver;
  std::mt19937 gen(5);
  std::normal_distribution<double> noise(0, 3);
  const double eps = double(std::numeric_limits<T>::epsilon());
  double worst = 0;
  for (int n : {2, 10, 100})
    for (int p : {1, 2})
      for (double alpha : {1e-3, 1., 50., 1e4}) {
        if (p >= n) continue;
        std::vector<T> v(n), r(n), exact(n);
        for (int j = 0; j < n; ++j) v[j] = T(noise(gen) + (j < p ? 5 : 0));
        sdca::detail::entropy_expansion<T> expansion(static_cast<std::size_t>(n));
        block_solver expanded(n, p, T(alpha), v.data(), r.data()),
            plain(n, p, T(alpha), v.data(), exact.data());
        expanded.expansion = &expansion;
        const T base = expanded.start();
        for (double s : {-1., -.5, -1e-3, 1e-6, .25, 1., -30., 30., 200., -200., 500., -500.}) {
          // Steps of the expansion (|s| <= 1), and of the warm evaluations from base.
          expansion.accurate = false;
          expanded.evaluate(base);
          const T dt = base + T(s) * block_solver::expansion_reach();
          if (std::abs(s) > 1 && std::abs(dt - base) > block_solver::warm_reach()) continue;
          const T f = expanded.evaluate(dt), g = plain.evaluate(dt);
          double allowed = 0;
          for (int j = 0; j < n; ++j) {
            // A label's residual solves to 8 eps of its argument in each.
            const double x = std::abs(double(v[j]) - double(plain.v_ref)) + std::abs(double(dt)) +
                             std::abs(std::log(alpha)) + std::log(double(p)),
                         prob = j < p ? 1. / p - double(exact[j]) : -double(exact[j]),
                         argument = j < p ? 16 * std::abs(double(plain.v_ref - v[j] + dt)) : 0,
                         tolerance = eps * (4 * std::abs(double(exact[j])) +
                                            (x + argument + 4) * prob / (1 + alpha * prob));
            allowed += tolerance;
            if (tolerance > 0)
              worst = std::max(worst, std::abs(double(r[j]) - double(exact[j])) / tolerance);
          }
          EXPECT_LE(std::abs(double(f) - double(g)), allowed)
              << precision << " n " << n << " p " << p << " alpha " << alpha << " step " << s;
        }
      }
  EXPECT_LE(worst, 1) << precision;
}

TEST(NumericalRegression, expanded_entropy_evaluations_match_exact_ones) {
  expanded_entropy_evaluations<double>("float64");
  expanded_entropy_evaluations<float>("float32");
}

TEST(NumericalRegression, warm_entropy_keeps_small_duals_beside_common_offset) {
  warm_entropy_common_offset<double, double>("float64");
  warm_entropy_common_offset<float, double>("float32 storage / float64 compute");
  warm_entropy_common_offset<float, float>("float32");
}

// Zero scores, n = 1000, C = float(1e35); the top-k (k = 2) caps are inactive.
// Oracle a0: log(a)-log(C-a)+(1+1/999)a = log(999), 100 digits.
template <typename Data, typename Result>
static void zero_score_high_c(std::size_t k, double tolerance) {
  const std::size_t n = 1000;
  const double a0 = 82.99536980534230771388;
  std::vector<Data> a(n, Data(0)), z(n, Data(0));
  const Result c = Result(1.0000000409184788e+35);
  if (k == 1)
    sdca::l2_entropy<Data, Result>(c).update_dual_variables(n, Data(1), a.data(), z.data());
  else
    sdca::l2_entropy_topk<Data, Result>(c, k).update_dual_variables(n, Data(1), a.data(), z.data());
  const double eps = double(std::numeric_limits<Data>::epsilon());
  EXPECT_LE(std::abs(double(a[0]) - a0), tolerance * eps * a0) << "k=" << k;
  EXPECT_LE(std::abs(double(a[n - 1]) + a0 / 999), tolerance * eps * a0) << "k=" << k;
}

TEST(NumericalRegression, zero_score_high_c_entropy_accuracy) {
  zero_score_high_c<float, double>(1, 2);
  zero_score_high_c<float, double>(2, 2);
  zero_score_high_c<double, double>(1, 64);
  zero_score_high_c<double, double>(2, 4096);
}
#endif

// Two-class hinge closed forms at the margin h = 1 + z_1 - z_0: the value
// h v - width v^2 / 2 and slope v = clip(h / width, 0, 1), or the nonsmooth
// max(0, h) and its slope (width 0). The smooth top-k width is gamma, the
// smooth multilabel width 2 gamma.
struct two_class_hinge {
  double value, slope;
  two_class_hinge(const double h, const double width) {
    if (width == 0) {
      value = std::max(0., h);
      slope = h > 0 ? 1 : 0;
    } else {
      slope = std::min(1., std::max(0., h / width));
      value = h * slope - width * slope * (slope / 2);
    }
  }
};

#ifndef __FAST_MATH__
TEST(NumericalRegression, hinge_tiny_margins_and_smoothing_overflow_losses) {
  const sdca::loss_kind kinds[] = {
      sdca::loss_kind::topk_hinge_alpha, sdca::loss_kind::topk_hinge_beta,
      sdca::loss_kind::topk_hinge_alpha, sdca::loss_kind::topk_hinge_beta,
      sdca::loss_kind::multilabel_hinge, sdca::loss_kind::multilabel_hinge};
  // {z_0, z_1, gamma, h}: a margin cancelling in 1 + z_1 - z_0, huge and tiny
  // smoothing (products overflow or underflow) and huge common offsets.
  const double cases[][4] = {{1, 1e-20, 1e-20, 1e-20}, {1, 1e-20, 1, 1e-20},
                             {0, 1e200, 1e300, 1e200}, {1, 1e-200, 1e-300, 1e-200},
                             {1e16, 1e16 + 2, 1, 3},   {1e16, 1e16 - 2, 1, -1},
                             {0, 0, 1e-300, 1},        {-1e16, -1e16 + 2, 1e16, 3}};
  for (const auto& c : cases)
    for (int i = 0; i < 6; ++i) {
      const bool smooth = i == 2 || i == 3 || i == 5;
      const two_class_hinge expected(c[3], smooth ? (i == 5 ? 2 : 1) * c[2] : 0);
      double z[] = {c[0], c[1]}, g[2];
      const double value = sdca::loss_and_gradient(kinds[i], 2, 1, z, g, 1, smooth ? c[2] : 0);
      // Errors relative to the value and to the slope (zero when inactive).
      const double scale = std::max(std::abs(expected.value), 1e-300);
      const double slope = std::max(expected.slope, 1e-300);
      EXPECT_NEAR(value / scale, expected.value / scale, 1e-15) << i << " " << c[1] << " " << c[2];
      EXPECT_NEAR(g[0] / slope, -expected.slope / slope, 1e-15) << i << " " << c[1] << " " << c[2];
      EXPECT_NEAR(g[1] / slope, expected.slope / slope, 1e-14) << i << " " << c[1] << " " << c[2];
    }
  // Represented float scores: the margin 2^-26 is lost to 1 + z_1 - z_0 in float.
  float zf[] = {1, std::ldexp(1.f, -26)}, gf[2];
  for (int i = 0; i < 6; ++i) {
    const bool smooth = i == 2 || i == 3 || i == 5;
    const two_class_hinge expected(std::ldexp(1., -26), smooth ? (i == 5 ? 2 : 1) : 0);
    const double value = sdca::loss_and_gradient(kinds[i], 2, 1, zf, gf, 1, smooth ? 1 : 0);
    EXPECT_NEAR(value / expected.value, 1, 1e-14) << i;
    EXPECT_NEAR(double(gf[1]) / expected.slope, 1, 1e-7) << i;
  }
}

TEST(NumericalRegression, float_losses_keep_stable_formulas_and_fall_back) {
  // Float arithmetic keeps the compensated margin 2^-26 and tiny smoothing.
  const float tiny = std::ldexp(1.f, -26);
  const float cases[][3] = {{1, tiny, 1}, {1, 1e-20f, 1e-20f}, {1, 1e-20f, 1}};
  for (const auto& c : cases)
    for (int i = 2; i < 8; ++i) {
      const bool smooth = i == 4 || i == 5 || i == 7;
      const two_class_hinge expected(double(c[1]), smooth ? (i == 7 ? 2 : 1) * double(c[2]) : 0);
      float z[] = {c[0], c[1]}, g[2];
      const double value = sdca::loss_and_gradient<float, float>(loss_case(i).first, 2, 1, z, g, 1,
                                                                 smooth ? double(c[2]) : 0);
      // A value below the normal floats (h^2 / 2 ~ 5e-41) keeps only subnormal resolution.
      const double floor = 4 * double(std::numeric_limits<float>::denorm_min());
      EXPECT_NEAR(value, expected.value, 1e-6 * expected.value + floor)
          << i << " " << c[1] << " " << c[2];
      EXPECT_NEAR(double(g[1]) / expected.slope, 1, 1e-6) << i << " " << c[1] << " " << c[2];
    }
  // Ordinary scores: float arithmetic is close to double; a spread or alpha
  // float cannot hold computes in double.
  std::mt19937 gen(5);
  std::normal_distribution<float> normal(0, 3);
  for (int i = 0; i < 9; ++i) {
    std::vector<float> z(12), g32(12), g64(12);
    for (auto& x : z) x = normal(gen);
    const std::size_t p = i == 6 || i == 7 ? 3 : 1;
    const auto kind = loss_case(i).first;
    const double gamma = loss_case(i).second ? 1 : 0;
    const double v32 =
        sdca::loss_and_gradient<float, float>(kind, 12, p, z.data(), g32.data(), 3, gamma);
    const double v64 =
        sdca::loss_and_gradient<float, double>(kind, 12, p, z.data(), g64.data(), 3, gamma);
    EXPECT_NEAR(v32, v64, 1e-5 * std::max(1., std::abs(v64))) << i;
    for (std::size_t j = 0; j < 12; ++j) EXPECT_NEAR(g32[j], g64[j], 1e-5) << i << " " << j;
    z[1] = 3e38f;
    z[2] = -3e38f;
    for (double alpha : {1., 1e-300}) {
      const double w32 =
          sdca::loss_and_gradient<float, float>(kind, 12, p, z.data(), g32.data(), 3, gamma, alpha);
      const double w64 = sdca::loss_and_gradient<float, double>(kind, 12, p, z.data(), g64.data(),
                                                                3, gamma, alpha);
      EXPECT_EQ(w32, w64) << i;
      EXPECT_EQ(g32, g64) << i;
    }
  }
}

TEST(NumericalRegression, float_topk_hinge_sums_many_margins_accurately) {
  // Margins 1 + z_j in [1, 2), exact in float: summed in float, the k of them
  // drifted by up to 1.4e-5 relative here.
  const std::size_t n = 100001;
  std::mt19937 gen(7);
  std::vector<float> z(n), g32(n), g64(n);
  z[0] = 0;
  for (std::size_t j = 1; j < n; ++j) z[j] = std::ldexp(float(gen() >> 9), -23);
  for (auto kind : {sdca::loss_kind::topk_hinge_alpha, sdca::loss_kind::topk_hinge_beta})
    for (std::size_t k : {65UL, 1000UL, 50000UL, n - 1}) {
      const double v32 = sdca::loss_and_gradient<float, float>(kind, n, 1, z.data(), g32.data(), k);
      const double v64 =
          sdca::loss_and_gradient<float, double>(kind, n, 1, z.data(), g64.data(), k);
      EXPECT_NEAR(v32 / v64, 1, 1e-7) << int(kind) << " " << k;
      EXPECT_EQ(g32, g64) << int(kind) << " " << k;
    }
}

TEST(NumericalRegression, topk_hinge_selects_the_stable_sort_top_k_on_ties) {
  // Integer scores tie heavily; both selections (partial_sort up to
  // max(16, m / 32), nth_element above) must pick the first k in stable order.
  std::mt19937 gen(3);
  for (std::size_t n : {40UL, 700UL, 5000UL}) {
    std::vector<double> z(n), g(n);
    for (auto& x : z) x = double(gen() % 5);
    std::vector<std::size_t> order(n - 1);
    std::iota(order.begin(), order.end(), std::size_t(1));
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t i, std::size_t j) { return z[i] > z[j]; });
    for (std::size_t k = 2; k < n; k += 1 + k / 8) {
      const double value =
          sdca::loss_and_gradient(sdca::loss_kind::topk_hinge_alpha, n, 1, z.data(), g.data(), k);
      std::vector<double> expected(n, 0.);
      double sum = 0;
      for (std::size_t j = 0; j < k; ++j) sum += 1 + z[order[j]] - z[0];
      if (sum > 0)
        for (std::size_t j = 0; j < k; ++j) expected[order[j]] = 1. / double(k);
      expected[0] = sum > 0 ? -1. : 0.;
      EXPECT_NEAR(value, std::max(0., sum / double(k)), 1e-12) << n << " " << k;
      for (std::size_t j = 0; j < n; ++j) EXPECT_NEAR(g[j], expected[j], 1e-12) << n << " " << k;
    }
  }
}

TEST(NumericalRegression, smooth_hinge_primal_losses_without_overflow) {
  using topk_smooth = sdca::l2_topk_hinge_smooth<double, double>;
  using hinge_smooth = sdca::l2_hinge_topk_smooth<double, double>;
  using multilabel_smooth = sdca::l2_multilabel_hinge_smooth<double, double>;
  double z[] = {0, 1e200}, y[] = {0, 1e200}, x[] = {0, 1e200};
  const double topk = two_class_hinge(1e200, 1e300).value;
  EXPECT_NEAR(hinge_smooth(1, 1, 1e300).primal_loss(2, z) / topk, 1, 1e-14);
  EXPECT_NEAR(topk_smooth(1, 1, 1e300).primal_loss(2, y) / topk, 1, 1e-14);
  const double multilabel = two_class_hinge(1e200, 2e300).value;
  EXPECT_NEAR(multilabel_smooth(1, 1e300).primal_loss(2, 1, x) / multilabel, 1, 1e-14);
  for (double margin : {1e-20, std::ldexp(1., -26)}) {
    double t[] = {1, margin}, u[] = {1, margin}, w[] = {1, margin};
    const double smooth = two_class_hinge(margin, 1e-20).value;
    EXPECT_NEAR(topk_smooth(1, 1, 1e-20).primal_loss(2, t) / smooth, 1, 1e-14);
    EXPECT_NEAR((sdca::l2_hinge_topk<double, double>(1, 1).primal_loss(2, u)) / margin, 1, 1e-14);
    const double pair = two_class_hinge(margin, 2e-20).value;
    EXPECT_NEAR(multilabel_smooth(1, 1e-20).primal_loss(2, 1, w) / pair, 1, 1e-14);
  }
}

// One cold two-class step from zero duals; the block is [s, -s] with
// s = h / (2 norm2 + w / C) below C, w = 0 for the nonsmooth hinges, gamma for
// smooth top-k and 2 gamma for smooth multilabel. The step forms the margin
// h = 1 + z1 - z0 to about an ulp of its terms (absolute).
template <typename Data, typename Result>
void check_two_class_steps(const Data z0, const Data z1, const double h, const double norm2) {
  const double C = 4, gamma = .5;
  auto check = [&](const char* name, const double width, Data* a) {
    const double den = 2 * double(Data(norm2)) + width / C, s = h / den;
    const double terms = 1 + std::abs(double(z1) - double(z0));
    const double tol = 4 * double(std::numeric_limits<Data>::epsilon()) * (s + terms / den);
    EXPECT_NEAR(double(a[0]), s, tol) << name << " " << sdca::type_name<Result>() << " " << h;
    EXPECT_NEAR(double(a[1]), -s, tol) << name << " " << sdca::type_name<Result>() << " " << h;
  };
  {
    Data z[] = {z0, z1}, a[] = {0, 0};
    sdca::l2_hinge_topk<Data, Result>(C, 1).update_dual_variables(2, Data(norm2), a, z);
    check("topk_hinge_alpha", 0, a);
  }
  {
    Data z[] = {z0, z1}, a[] = {0, 0};
    sdca::l2_hinge_topk_smooth<Data, Result>(C, 1, gamma)
        .update_dual_variables(2, Data(norm2), a, z);
    check("topk_hinge_alpha", gamma, a);
  }
  {
    Data z[] = {z0, z1}, a[] = {0, 0};
    sdca::l2_topk_hinge<Data, Result>(C, 1).update_dual_variables(2, Data(norm2), a, z);
    check("topk_hinge_beta", 0, a);
    Data y[] = {z0, z1}, b[] = {0, 0};
    sdca::l2_topk_hinge_smooth<Data, Result>(C, 1, gamma)
        .update_dual_variables(2, Data(norm2), b, y);
    check("topk_hinge_beta", gamma, b);
  }
  {
    Data z[] = {z0, z1}, a[] = {0, 0};
    sdca::l2_multilabel_hinge<Data, Result>(C).update_dual_variables(2, 1, Data(norm2), a, z);
    check("multilabel_hinge", 0, a);
  }
  {
    Data z[] = {z0, z1}, a[] = {0, 0};
    sdca::l2_multilabel_hinge_smooth<Data, Result>(C, gamma).update_dual_variables(
        2, 1, Data(norm2), a, z);
    check("multilabel_hinge", 2 * gamma, a);
  }
}

TEST(NumericalRegression, hinge_steps_resolve_margins_in_every_precision) {
  const float tiny = std::ldexp(1.f, -26);
  for (double norm2 : {1., .7, 3e4}) {
    check_two_class_steps<float, float>(1, tiny, double(tiny), norm2);
    check_two_class_steps<float, double>(1, tiny, double(tiny), norm2);
    check_two_class_steps<double, double>(1, double(tiny), double(tiny), norm2);
    // The margin of 1 + 1e-20 - 1 cancels to zero.
    check_two_class_steps<float, float>(1, 1e-20f, double(1e-20f), norm2);
    check_two_class_steps<float, double>(1, 1e-20f, double(1e-20f), norm2);
    check_two_class_steps<double, double>(1, 1e-20, 1e-20, norm2);
    // Huge common offsets keep the margin: z_1 - z_0 is exact.
    check_two_class_steps<double, double>(1e16, 1e16 + 2, 3, norm2);
    check_two_class_steps<double, double>(-1e16, -1e16 + 2, 3, norm2);
    check_two_class_steps<float, double>(1e16f, 1e16f + 2e9f,
                                         1 + double(1e16f + 2e9f) - double(1e16f), norm2 * 1e9);
  }
}
#endif

// Multilabel hinge steps and smoothed losses agree to rounding in both
// projection precisions on ordinary blocks.
TEST(NumericalRegression, multilabel_hinge_projection_precisions_agree) {
  std::mt19937 gen(23);
  std::normal_distribution<double> d(0, 1);
  const double eps = std::numeric_limits<double>::epsilon();
  const auto f64 = sdca::projection_precision::float64, ext = sdca::projection_precision::extended;
  for (int i = 0; i < 1000; ++i) {
    const std::size_t n = 3 + std::size_t(i) % 30, p = 1 + std::size_t(i) % 2;
    const double c = std::ldexp(1., i % 7 - 3);
    std::vector<double> z(n), a(n, 0.);
    for (auto& x : z) x = d(gen);
    std::vector<double> z1 = z, z2 = z, a1 = a, a2 = a;
    sdca::l2_multilabel_hinge_smooth<double, double> obj(c, 1);
    obj.update_dual_variables(n, p, 1., a1.data(), z1.data(), f64);
    obj.update_dual_variables(n, p, 1., a2.data(), z2.data(), ext);
    double scale = c;
    for (double x : z) scale = std::max(scale, std::abs(x));
    for (std::size_t j = 0; j < n; ++j) EXPECT_NEAR(a1[j], a2[j], 16 * eps * scale) << i;
    std::vector<double> g1(n), g2(n);
    const double v1 = sdca::loss_and_gradient(sdca::loss_kind::multilabel_hinge, n, p, z.data(),
                                              g1.data(), 1, 1., 1., f64);
    const double v2 = sdca::loss_and_gradient(sdca::loss_kind::multilabel_hinge, n, p, z.data(),
                                              g2.data(), 1, 1., 1., ext);
    EXPECT_NEAR(v1, v2, 16 * eps * std::max(1., std::abs(v2))) << i;
  }
}

// Far offsets and tiny smoothing: the projected mass is small next to the
// scores, so float64 must hold its outputs relative to a score origin, as
// extended does (labels first; rows from the Python exact-reference tests).
TEST(NumericalRegression, multilabel_hinge_far_offsets_match_extended) {
  const double eps = std::numeric_limits<double>::epsilon();
  auto check = [&](std::vector<double> z, std::size_t p, double gamma) {
    const std::size_t n = z.size();
    std::vector<double> g1(n), g2(n);
    const double v1 =
        sdca::loss_and_gradient(sdca::loss_kind::multilabel_hinge, n, p, z.data(), g1.data(), 1,
                                gamma, 1., sdca::projection_precision::float64);
    const double v2 =
        sdca::loss_and_gradient(sdca::loss_kind::multilabel_hinge, n, p, z.data(), g2.data(), 1,
                                gamma, 1., sdca::projection_precision::extended);
    EXPECT_NEAR(v1, v2, 16 * eps * std::abs(v2)) << z[0] << " " << gamma;
    for (std::size_t j = 0; j < n; ++j) EXPECT_NEAR(g1[j], g2[j], 16 * eps) << z[0] << " " << j;
  };
  check({700, 1, -.3, 0, 700, .2}, 3, 1e-3);
  check({-700, 1, -1, 0, 2}, 3, 1e-3);
  check({-1e30, 1, -1, 0, 2}, 3, 1e20);
  std::mt19937 gen(29);
  std::normal_distribution<double> d(0, 1);
  for (int k = 0; k <= 30; k += 3)
    for (double gamma : {1e-6, 1e-3, 1., 1e3})
      for (int i = 0; i < 20; ++i) {
        std::vector<double> z(6);
        for (auto& x : z) x = d(gen);
        z[i % 3] += (i % 2 ? -1 : 1) * std::pow(10., k);
        z[3 + i % 3] += std::pow(10., k);
        check(z, 3, gamma);
      }
  // Ties: integer scores in [-2, 2], tiny to unit smoothing, 3 to 20 classes.
  std::uniform_int_distribution<int> score(-2, 2);
  for (int i = 0; i < 2000; ++i) {
    const std::size_t n = 3 + std::size_t(i) % 18, p = 1 + std::size_t(i / 18) % (n - 1);
    std::vector<double> z(n);
    for (auto& x : z) x = double(score(gen));
    check(z, p, std::pow(10., -6 + i % 7));
  }
  check({2, -2, -2, -2, -2, -2}, 4, 1e-3);
}

TEST(NumericalRegression, distillation_losses_and_solver_certify_gap) {
  const double s[] = {1.5, -0.5, 0.2}, t[] = {1.5, -0.5, 0.2};
  double g[3] = {1., 1., 1.}, pg[3] = {1., 1., 1.};
  const double v = sdca::distillation_loss_and_gradient(3, s, t, g, 2.0);
  const double pv = sdca::proximal_distillation_loss_and_gradient(3, s, t, pg, 2.0, 0.5);
  EXPECT_NEAR(v, pv, 1e-14);
  for (int j = 0; j < 3; ++j) {
    EXPECT_NEAR(g[j], 0., 1e-15);
    EXPECT_NEAR(pg[j], 0., 1e-15);
  }
  // Column-major X (d=2, n=3): features of 3 examples.
  const double x[] = {1.0, 0.0, 0.0, 1.0, -1.0, -1.0};
  const double q[] = {0.8, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.8};
  double w[6] = {0}, a[9] = {0};
  auto ctx =
      sdca::make_context(sdca::make_input_feature(2, 3, x), sdca::make_output_distribution(3, 3, q),
                         sdca::make_objective_l2_distillation<double>(2.0), a, w);
  ctx.criteria.epsilon = 1e-8;
  sdca::make_solver(ctx).solve();
  EXPECT_EQ(ctx.status, sdca::solver_status::solved);
  EXPECT_LE(ctx.train.evals.back().relative_gap(), 1e-8);
}

#endif  // SDCA_TESTS_REGRESSION_SOLVER_H
