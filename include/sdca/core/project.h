// The projections and proximal maps of the losses and of the solver's dual steps (no include
// guard: each header guards itself once per instruction set, sdca/core/isa.h).

#include "sdca/core/project_base.h"
#include "sdca/core/project_entropy.h"
#include "sdca/core/project_knapsack.h"
#include "sdca/core/project_softmax.h"
#include "sdca/core/project_topk.h"
#include "sdca/core/project_topk_entropy.h"
#include "sdca/core/project_two_simplex.h"
