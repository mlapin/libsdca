rng(0);

%%%
%%% Test prox
%%%
if 1
  d = 10;
  n = 10;

  opts.prox = 'knapsack';

  A = randn(d,n) + 10*randn(d,n) + 100*randn(d,n) + 1000*randn(d,n);

  B = matsdca_prox(A, opts);

end
