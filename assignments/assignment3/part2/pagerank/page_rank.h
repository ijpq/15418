#ifndef __PAGE_RANK_H__
#define __PAGE_RANK_H__

#include "graph_dist.h"
#include "graph_dist_ref.h"

void pageRank_ref(DistGraphRef &g, double* solution, double damping, double convergence);

void pageRank(DistGraph &g, double* solution, double damping, double convergence);

#endif /* __PAGE_RANK_H__ */