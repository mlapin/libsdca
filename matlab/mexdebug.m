rng(0);

%%%
%%% Test prox
%%%
if 0
  d = 10;
  n = 10;

  opts.prox = 'knapsack';

  A = randn(d,n) + 10*randn(d,n) + 100*randn(d,n) + 1000*randn(d,n);

  B = matsdca_prox(A, opts);

end

%%%
%%% Test solver
%%%
if 1
  load('data/indoor67-cnn-trn.mat', 'Xtrn', 'Ytrn');
  
  opts.objective = 'mlsoftmax';
  opts.eval_epoch = 1;
  opts.check_epoch = 1;
  opts.max_epoch = 20;
  opts.log_level = 'verbose';
  model = matsdca_fit(Xtrn, Ytrn, opts);
  disp(model);
  
  opts.objective = 'l2_entropy_topk';
  model2 = libsdca_solve(Xtrn, Ytrn, opts);
  
  disp(norm(model.A(:)-model2.A(:)));

end

