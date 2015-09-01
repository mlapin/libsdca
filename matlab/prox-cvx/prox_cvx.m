function [X,info] = prox_cvx(A, opts)
switch opts.prox
  case {'knapsack', 'knapsack_eq'}
    [X,info] = prox_cvx_knapsack_eq(A, opts);
  case 'knapsack_le'
    [X,info] = prox_cvx_knapsack_le(A, opts);
  case 'knapsack_le_biased'
    [X,info] = prox_cvx_knapsack_le_biased(A, opts);
  case 'topk_simplex'
    [X,info] = prox_cvx_topk_simplex(A, opts);
  case 'topk_simplex_biased'
    [X,info] = prox_cvx_topk_simplex_biased(A, opts);
  case 'topk_entropy'
    [X,info] = prox_cvx_topk_entropy(A, opts);
  case 'topk_entropy_biased'
    [X,info] = prox_cvx_topk_entropy_biased(A, opts);
  case 'entropy'
    [X,info] = prox_cvx_entropy(A, opts);
  case 'topk_cone'
    [X,info] = prox_cvx_topk_cone(A, opts);
  case 'topk_cone_biased'
    [X,info] = prox_cvx_topk_cone_biased(A, opts);
  otherwise
    fprintf('Unknown prox %s.\n', opts.prox);
    X = nan; info = [];
end
