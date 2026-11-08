#include "sdca/solver.h"
#include "test_util.h"

namespace {
// Reporting follows cout's format. Restore both formatting and logger state
// even when a test assertion fails, so numerical tests remain independent.
struct capture_log {
  std::ostringstream output;
  std::ostringstream saved;
  std::streambuf* buffer;
  sdca::logging::level level;
  capture_log() : buffer(std::cout.rdbuf()), level(sdca::logging::get_level()) {
    saved.copyfmt(std::cout);
    std::cout.copyfmt(output);
    std::cout.rdbuf(output.rdbuf());
  }
  ~capture_log() {
    std::cout.rdbuf(buffer);
    std::cout.copyfmt(saved);
    sdca::logging::set_level(level);
  }
};

void contains(const std::string& text, const std::string& expected) {
  EXPECT_NE(std::string::npos, text.find(expected)) << text;
}
}  // namespace

#ifndef __FAST_MATH__
TEST(ReportingTest, gaps_and_elapsed_times) {
  sdca::eval_train_base<double> e;
  EXPECT_EQ(0, e.absolute_gap());
  EXPECT_EQ(0, e.relative_gap());
  for (const auto& pair : {std::make_pair(10., 8.), std::make_pair(-8., -10.)}) {
    e.primal = pair.first;
    e.dual = pair.second;
    EXPECT_EQ(2, e.absolute_gap());
    EXPECT_DOUBLE_EQ(.2, e.relative_gap());
  }
  e.primal = 8;
  e.dual = 10;
  EXPECT_DOUBLE_EQ(-.2, e.relative_gap());
  for (double invalid :
       {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    e.primal = invalid;
    e.dual = 1;
    EXPECT_EQ(std::numeric_limits<double>::infinity(), e.relative_gap());
    e.primal = 1;
    e.dual = invalid;
    EXPECT_EQ(std::numeric_limits<double>::infinity(), e.relative_gap());
  }
  e.solve_time_cpu = 2;
  e.eval_time_cpu = 3;
  e.solve_time_wall = 4;
  e.eval_time_wall = 5;
  EXPECT_EQ(5, e.cpu_time());
  EXPECT_EQ(9, e.wall_time());
}
#endif

TEST(ReportingTest, train_metrics_and_optional_timing) {
  capture_log log;
  sdca::eval_train<double, sdca::multiclass_output> e;
  e.epoch = 7;
  e.primal = 10;
  e.dual = 8;
  e.primal_loss = 6;
  e.dual_loss = 11;
  e.primal_regularizer = 4;
  e.dual_regularizer = 3;
  e.solve_time_cpu = 2;
  e.eval_time_cpu = 3;
  e.solve_time_wall = 4;
  e.eval_time_wall = 5;
  e.accuracy = {.25, .5, .75, .875, .9375, .96875};
  EXPECT_EQ(.25, e.topk_accuracy(0));
  EXPECT_EQ(.96875, e.topk_accuracy(5));
  EXPECT_EQ(1, e.topk_accuracy(6));
  EXPECT_EQ(
      "epoch: 7, relative_gap: 0.2, absolute_gap: 2, primal: 10, dual: 8, "
      "primal_loss: 6, dual_loss: 11, primal_regularizer: 4, dual_regularizer: 3, "
      "accuracy: 0.25 0.5 0.75 0.875 0.9375 ",
      e.to_string(true));
  contains(e.to_string(), "cpu_time: 5 (solve: 2, eval: 3), wall_time: 9 (solve: 4, eval: 5)");
  e.accuracy.clear();
  EXPECT_EQ(1, e.topk_accuracy(0));
  contains(e.to_string(true), "accuracy: ");
  sdca::eval_train<double, sdca::multilabel_output> ml;
  ml.rank_loss = .125;
  contains(ml.to_string(true), "rank_loss: 0.125");
  EXPECT_EQ(std::string::npos, ml.to_string(true).find("cpu_time"));
  contains(ml.to_string(), "cpu_time: 0");
}

TEST(ReportingTest, test_metrics) {
  capture_log log;
  sdca::eval_test<double, sdca::multiclass_output> e;
  EXPECT_EQ(1, e.topk_accuracy(0));
  EXPECT_EQ("primal_loss: 0, accuracy: ", e.to_string());
  e.primal_loss = 2;
  e.accuracy = {.25, .5};
  EXPECT_EQ(.25, e.topk_accuracy(0));
  EXPECT_EQ(.5, e.topk_accuracy(1));
  EXPECT_EQ(1, e.topk_accuracy(2));
  EXPECT_EQ("primal_loss: 2, accuracy: 0.25 0.5 ", e.to_string());
  e.accuracy = {.25, .5, .75, .875, .9375, .96875};
  EXPECT_EQ(std::string::npos, e.to_string().find("0.96875"));
  sdca::eval_test<double, sdca::multilabel_output> ml;
  ml.primal_loss = 3;
  ml.rank_loss = .25;
  EXPECT_EQ("primal_loss: 3, rank_loss: 0.25", ml.to_string());
}

TEST(ReportingTest, status_names_and_stopping_options) {
  using sdca::solver_status;
  const std::pair<solver_status, const char*> cases[] = {
      {solver_status::none, "none"},
      {solver_status::solving, "solving"},
      {solver_status::solved, "solved"},
      {solver_status::no_progress, "no_progress"},
      {solver_status::max_epoch, "max_epoch"},
      {solver_status::max_cpu_time, "max_cpu_time"},
      {solver_status::max_wall_time, "max_wall_time"},
      {solver_status::failed, "failed"}};
  for (const auto& c : cases) EXPECT_EQ(c.second, sdca::solver_status_name(c.first));
  sdca::stopping_criteria c;
  EXPECT_EQ(
      "epsilon: 0.001, eval_epoch: 10, max_epoch: 1000, max_cpu_time: 0, "
      "max_wall_time: 0, eval_on_start: 0",
      c.to_string());
  c.epsilon = .25;
  c.eval_epoch = 2;
  c.max_epoch = 7;
  c.max_cpu_time = 3;
  c.max_wall_time = 4;
  c.eval_on_start = true;
  EXPECT_EQ(
      "epsilon: 0.25, eval_epoch: 2, max_epoch: 7, max_cpu_time: 3, "
      "max_wall_time: 4, eval_on_start: 1",
      c.to_string());
}

TEST(ReportingTest, context_and_log_levels) {
  capture_log log;
  double x[] = {1, 2}, a[4] = {}, w[2] = {};
  std::vector<sdca::size_type> labels = {0, 1};
  auto ctx =
      sdca::make_context(sdca::make_input_feature(1, 2, x), sdca::multiclass_output(2, labels),
                         sdca::make_objective_l2_entropy<double>(), a, w);
  EXPECT_EQ("none", ctx.status_name());
  EXPECT_FALSE(ctx.is_dual());
  EXPECT_FALSE(ctx.is_prox());
  contains(ctx.status_string(), "status: none, cpu_time: 0");
  contains(ctx.to_string(), "features (num_dimensions: 1, num_examples: 2, precision: double)");
  contains(ctx.to_string(), "is_prox: 0");
  ctx.primal_initial = w;
  EXPECT_TRUE(ctx.is_prox());
  contains(ctx.to_string(), "is_prox: 1");
  ctx.solve_time.cpu.elapsed = 2;
  ctx.eval_time.cpu.elapsed = 3;
  ctx.solve_time.wall.elapsed = 4;
  ctx.eval_time.wall.elapsed = 5;
  EXPECT_EQ(5, ctx.cpu_time());
  EXPECT_EQ(5, ctx.cpu_time_now());
  EXPECT_EQ(9, ctx.wall_time());
  EXPECT_EQ(9, ctx.wall_time_now());
  ctx.train.evals.emplace_back();
  ctx.train.evals.back().epoch = 7;
  ctx.status = sdca::solver_status::solved;
  contains(ctx.status_string(), "status: solved, epoch: 7");
  contains(ctx.status_string(),
           "cpu_time: 5 (solve: 2, eval: 3), wall_time: 9 (solve: 4, eval: 5)");

  sdca::logging::set_level(sdca::logging::level::none);
  sdca::reporting::begin_solve(ctx);
  sdca::reporting::end_solve(ctx);
  EXPECT_TRUE(log.output.str().empty());
  sdca::logging::set_level(sdca::logging::level::debug);
  sdca::reporting::begin_solve(ctx);
  sdca::reporting::end_solve(ctx);
  contains(log.output.str(), "Solve: ");
  contains(log.output.str(), "Solution: status: solved");
  log.output.str("");
  ctx.epoch = 9;
  sdca::reporting::end_epoch(ctx, true);
  EXPECT_TRUE(log.output.str().empty());
  sdca::reporting::end_epoch(ctx, false);
  contains(log.output.str(), "epoch: 9, cpu_time: 5 (2 + 3), wall_time: 9 (4 + 5)");
  log.output.str("");
  sdca::reporting::eval_created(ctx.train.evals.back(), 0);
  contains(log.output.str(), "epoch: 7");
  sdca::eval_test<double, sdca::multiclass_output> test;
  sdca::reporting::eval_created(test, 2);
  contains(log.output.str(), "eval on set #2: primal_loss: 0");
  log.output.str("");
  sdca::logging::set_level(sdca::logging::level::warning);
  ctx.status = sdca::solver_status::max_epoch;
  sdca::reporting::end_solve(ctx);
  contains(log.output.str(), "Solve: ");
  contains(log.output.str(), "Solution: status: max_epoch");
  sdca::reporting::solver_stop_failed(-.5, .001, .01);
  contains(log.output.str(),
           "negative duality gap; absolute_gap: -0.5, eps_machine: 0.001, eps_user: 0.01");
  auto before = ctx.train.evals.back();
  before.dual = 2;
  ctx.train.evals.back().dual = 1;
  sdca::reporting::solver_stop_no_progress(ctx.train.evals.back(), before);
  contains(log.output.str(), "dual objective decreased; dual: 1, dual_before: 2, difference: -1");
}
