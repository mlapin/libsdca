#ifndef SDCA_SOLVER_H
#define SDCA_SOLVER_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

#include "sdca/solver/context.h"
#include "sdca/solver/eval.h"
#include "sdca/solver/update.h"

namespace sdca {

namespace detail {

// Gap-safe screening needs W and single labels.
template <typename Input, typename Output>
inline bool screens_input(const Input&, const Output&) {
  return false;
}
template <typename Data>
inline bool screens_input(const feature_input<Data>&, const multiclass_output&) {
  return true;
}

template <typename Context, typename Input, typename Output>
inline void screen_input(const Context&, const Input&, const Output&, const screening_rules&,
                         const std::vector<double>&, const double, std::vector<char>&,
                         std::vector<char>&) {}
template <typename Context, typename Data>
inline void screen_input(const Context& ctx, const feature_input<Data>& in,
                         const multiclass_output& out, const screening_rules& rules,
                         const std::vector<double>& norms, const double gap,
                         std::vector<char>& example, std::vector<char>& klass) {
  screen_examples(rules, out.num_examples(), in.num_dimensions, out, in.features,
                  ctx.primal_variables, norms, gap, ctx.criteria.num_threads, example, klass);
}

template <typename Input>
inline std::vector<double> input_norms(const Input&, const std::size_t) {
  return std::vector<double>();
}
template <typename Data>
inline std::vector<double> input_norms(const feature_input<Data>& in,
                                       const std::size_t num_threads) {
  return example_norms<double>(in.num_dimensions, in.num_examples, in.features, num_threads);
}

}  // namespace detail

// SDCA on a solver_context: passes over the training examples in a random
// order, updating each dual block and W, until ctx.criteria stops it. The
// hinge objectives shrink (stopping_criteria::shrinking): between full passes
// (each scheduled evaluation's epoch) the heuristic skips the examples whose last step left
// their block unchanged and steps a single-label block on its classes of
// nonzero dual; the safe screening leaves out for good the zero duals that
// the certified gap proves zero at the optimum. Every evaluation covers all
// examples and classes.
template <typename Data, typename Result, template <typename> class Input, typename Output,
          template <typename, typename> class Objective>
class solver {
 public:
  typedef Data data_type;
  typedef Result result_type;
  typedef Input<Data> input_type;
  typedef Output output_type;
  typedef Objective<Data, Result> objective_type;

  typedef solver_context<Data, Result, Input, Output, Objective> context_type;
  typedef typename context_type::variable_type variable_type;
  typedef solver_scratch<variable_type, Input> scratch_type;

  explicit solver(context_type& context_) : ctx_(context_), is_evaluated_(false) {}

  // Runs epochs from ctx.epoch until a stopping criterion holds; ctx.status
  // gives the reason. A failed step stops it with solver_status::failed.
  void solve() {
    begin_solve();
    try {
      while (ctx_.status == solver_status::solving) {
        begin_epoch();
        const bool full = !heuristic_ || is_full_pass();
        const size_type m = ctx_.train.num_classes();
        size_type steps = 0, changed = 0;
        for (auto& example : examples_) {
          if ((!full && shrunk_[example]) || (safe_ && screened_example_[example] &&
                                              is_zero_block(ctx_.dual_variables + m * example))) {
            ++num_skipped_;
            continue;
          }
          const char* screened = safe_ ? &screened_class_[m * example] : nullptr;
          shrunk_[example] = update_variables(example, ctx_, scratch_, full, screened);
          ++steps;
          changed += !shrunk_[example];
        }
        num_updates_ += steps;
        steps_ += full ? ctx_.train.num_examples() : steps;
        idle_ = !full && changed == 0;

        end_epoch();
      }
    } catch (const std::exception& e) {
      // The failed step restored its block, so the iterate stays consistent.
      ctx_.status = solver_status::failed;
      reporting::solver_stop_step_failed(e);
    }
    end_solve();
  }

  // The coordinate steps taken and skipped by shrinking in the last solve,
  // and the examples whose whole block screening fixed at zero.
  size_type num_updates() const { return num_updates_; }
  size_type num_skipped() const { return num_skipped_; }
  size_type num_screened() const {
    return static_cast<size_type>(
        std::count(screened_example_.begin(), screened_example_.end(), char(1)));
  }
  // The classes that screening fixed at zero, over all examples.
  size_type num_screened_classes() const {
    return static_cast<size_type>(
        std::count(screened_class_.begin(), screened_class_.end(), char(1)));
  }
  // The marks of screening: per example, then per class of each example (n by m).
  const std::vector<char>& screened_examples() const { return screened_example_; }
  const std::vector<char>& screened_classes() const { return screened_class_; }

 protected:
  context_type& ctx_;
  scratch_type scratch_;

  bool is_evaluated_;
  std::minstd_rand generator_;
  std::vector<size_type> examples_;
  // Per example, whether its last step left the block unchanged; epochs
  // between full passes skip such examples.
  std::vector<char> shrunk_;
  size_type first_epoch_ = 0;
  // The steps since the last evaluation (a full pass counts every example),
  // its epoch, whether this epoch ends with a scheduled evaluation, and
  // whether the last epoch between full passes changed no block.
  size_type steps_ = 0, eval_epoch_ = 0;
  bool scheduled_ = false, idle_ = false;
  size_type num_updates_ = 0;
  size_type num_skipped_ = 0;
  // Safe screening: the marks of screen_examples, the example norms, and the
  // certified gap of the last screening.
  bool heuristic_ = false, safe_ = false;
  screening_rules rules_;
  std::vector<char> screened_example_, screened_class_;
  std::vector<double> norms_;
  double screen_gap_ = 0, screen_factor_ = 2;
  // Evaluations left ahead of the schedule, and the epoch of a zero dual
  // start (relative gap one) if the solve has one.
  size_type early_evals_ = 0;
  size_type zero_epoch_ = 0;
  bool zero_start_ = false;

  void begin_solve() {
    reporting::begin_solve(ctx_);
    ctx_.load_variables();

    ctx_.status =
        (ctx_.criteria.max_epoch > ctx_.epoch) ? solver_status::solving : solver_status::max_epoch;

    // Within max_epoch / eval_epoch + 2 evaluations, the start and the end included.
    const auto& criteria = ctx_.criteria;
    const size_type spare = criteria.eval_epoch > 0 && criteria.epsilon > 0
                                ? (criteria.max_epoch % criteria.eval_epoch == 0 ? 2 : 1)
                                : 0;
    early_evals_ = spare > size_type(criteria.eval_on_start) ? spare - criteria.eval_on_start : 0;
    const size_type duals = ctx_.train.num_examples() * ctx_.train.num_classes();
    zero_start_ = std::all_of(ctx_.dual_variables, ctx_.dual_variables + duals,
                              [](const variable_type a) { return a == 0; });
    zero_epoch_ = ctx_.epoch;

    scratch_.init(ctx_);
    begin_shrinking();
    if (ctx_.criteria.eval_on_start) {
      evaluate_solution();
      check_stopping_criteria<Data, Result>(ctx_);
      screen();
    }

    if (ctx_.status == solver_status::solving) {
      generator_.seed();
      examples_.resize(ctx_.train.num_examples());
      std::iota(examples_.begin(), examples_.end(), 0);
      first_epoch_ = ctx_.epoch;

      ctx_.solve_time.resume();
    }
  }

  void end_solve() {
    ctx_.solve_time.stop();
    if (!is_evaluated_) {
      evaluate_solution();
    }

    reporting::end_solve(ctx_);
  }

  // The modes of criteria.shrinking that the objective and the input take.
  void begin_shrinking() {
    shrinking_mode mode = ctx_.criteria.shrinking;
    if (mode == shrinking_mode::automatic) mode = shrinking_mode::both;
    const bool shrinks = detail::shrinks<objective_type>::value;
    heuristic_ = shrinks && (mode == shrinking_mode::heuristic || mode == shrinking_mode::both);
    rules_ = screening_rules_of(ctx_.objective);
    safe_ = shrinks && (mode == shrinking_mode::safe || mode == shrinking_mode::both) &&
            (rules_.examples || rules_.classes) &&
            detail::screens_input(ctx_.train.in, ctx_.train.out);
    const size_type n = ctx_.train.num_examples(), m = ctx_.train.num_classes();
    shrunk_.assign(n, 0);
    num_updates_ = num_skipped_ = steps_ = 0;
    idle_ = false;
    eval_epoch_ = ctx_.epoch;
    screened_example_.assign(safe_ ? n : 0, 0);
    screened_class_.assign(safe_ ? n * m : 0, 0);
    norms_ = safe_ ? detail::input_norms(ctx_.train.in, ctx_.criteria.num_threads)
                   : std::vector<double>();
    screen_gap_ = std::numeric_limits<double>::infinity();
    screen_factor_ = 2;
  }

  bool is_zero_block(const variable_type* block) const {
    return std::all_of(block, block + ctx_.train.num_classes(),
                       [](const variable_type a) { return a == 0; });
  }

  // After an evaluation that divides the certified gap of the last screening
  // by screen_factor_, marks the duals that the gap proves zero at the
  // optimum. A screening that marks nothing doubles the factor (from 2).
  void screen() {
    if (!safe_ || !is_evaluated_ || ctx_.status != solver_status::solving) return;
    const auto& e = ctx_.train.evals.back();
    const double gap = double(e.primal) - double(e.dual) + double(e.gap_error);
    if (!(gap >= 0) || !std::isfinite(gap) || !(gap * screen_factor_ <= screen_gap_)) return;
    screen_gap_ = gap;
    ctx_.eval_time.resume();
    const size_type before = num_screened_classes();
    // The gap is that of the stored iterate, so the margins are too.
    detail::screen_input(ctx_.stored(), ctx_.train.in, ctx_.train.out, rules_, norms_, gap,
                         screened_example_, screened_class_);
    screen_factor_ = num_screened_classes() > before ? 2 : 2 * screen_factor_;
    ctx_.eval_time.stop();
  }

  void begin_epoch() {
    is_evaluated_ = false;
    scheduled_ = is_scheduled();
    std::shuffle(examples_.begin(), examples_.end(), generator_);
  }

  // Whether this epoch ends at a multiple of eval_epoch with an evaluation.
  // Under the heuristic the epochs between full passes are cheap and the
  // evaluations would dominate: one is due once the epochs since the last
  // took as many steps as eval_epoch full passes, or full_pass_epoch epochs,
  // and at max_epoch, so that a fit solved by then reports it.
  bool is_scheduled() const {
    const auto& criteria = ctx_.criteria;
    const size_type next = ctx_.epoch + 1, n = std::max<size_type>(1, ctx_.train.num_examples());
    if (criteria.eval_epoch == 0 || next % criteria.eval_epoch != 0) return false;
    return !heuristic_ || next >= criteria.max_epoch || steps_ / n + 1 >= criteria.eval_epoch ||
           next - eval_epoch_ >= criteria.full_pass_epoch;
  }

  // The first epoch of a solve, the epochs of the scheduled evaluations and
  // the epoch after one between full passes that changed nothing update every
  // example (without evaluations, also every full_pass_epoch-th).
  bool is_full_pass() const {
    const size_type every = ctx_.criteria.full_pass_epoch;
    return every <= 1 || ctx_.epoch == first_epoch_ || scheduled_ || idle_ ||
           (ctx_.criteria.eval_epoch == 0 && (ctx_.epoch + 1) % every == 0);
  }

  void end_epoch() {
    ctx_.solve_time.stop();

    ++ctx_.epoch;
    if (scheduled_) {
      evaluate_solution();
    } else if (early_evals_ > 0 && predicts_solved()) {
      --early_evals_;
      evaluate_solution();
    }

    check_stopping_criteria<Data, Result>(ctx_);
    screen();

    reporting::end_epoch(ctx_, is_evaluated_);

    ctx_.solve_time.resume();
  }

  // Whether the certified relative gap, decreasing at the rate between the
  // last two evaluations, meets epsilon by this epoch. With one evaluation of
  // a zero dual start, the start counts as relative gap one: an assumption
  // (it holds where the primal there is positive), used only for the trend.
  bool predicts_solved() const {
    const auto& evals = ctx_.train.evals;
    if (evals.empty() || (evals.size() < 2 && !zero_start_)) return false;
    auto certified = [](const typename context_type::train_eval_type& e) {
      const double max = std::max(std::abs(double(e.primal)), std::abs(double(e.dual)));
      return (double(e.primal) - double(e.dual) + double(e.gap_error)) / max;
    };
    const auto& last = evals.back();
    const double now = certified(last),
                 before = evals.size() > 1 ? certified(evals.rbegin()[1]) : 1;
    const size_type from = evals.size() > 1 ? evals.rbegin()[1].epoch : zero_epoch_;
    if (!(now > 0 && now < before) || !(last.epoch > from) || !std::isfinite(before)) return false;
    const double rate = std::log(now / before) / double(last.epoch - from);
    return std::log(now) + rate * double(ctx_.epoch - last.epoch) <=
           std::log(ctx_.criteria.epsilon);
  }

  // Stores the iterate to the caller's buffers and evaluates them: the values
  // a solve returns (the last evaluation is that of the end).
  void evaluate_solution() {
    ctx_.eval_time.resume();
    ctx_.store_variables();
    const auto stored = ctx_.stored();

    evaluate_dataset(stored, ctx_.train, scratch_);

    size_type id(0);
    for (auto& test_set : ctx_.test) {
      evaluate_dataset(stored, test_set, scratch_, id++);
    }

    is_evaluated_ = true;
    steps_ = 0;
    eval_epoch_ = ctx_.epoch;
    ctx_.eval_time.stop();
  }
};

// The solver of a context (see make_context); the context must outlive it.
template <typename Data, typename Result, template <typename> class Input, typename Output,
          template <typename, typename> class Objective>
inline solver<Data, Result, Input, Output, Objective> make_solver(
    solver_context<Data, Result, Input, Output, Objective>& ctx) {
  return solver<Data, Result, Input, Output, Objective>(ctx);
}

}  // namespace sdca

#endif
