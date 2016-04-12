%
% Sample script for testing matsdca mex files
%
clear;
rng(0);

% Test prox
if 1
  d = 100;
  n = 30;

  A = randn(d,n);

  opts.prox = 'knapsack';
  B = matsdca_prox(A, opts);

end


% Test solver
if 1
  num_trn = 200;
  num_tst = 200000;
  num_val = 200;

  [Xtrn,Ytrn,Xval,Yval,Xtst,Ytst] = getdata(num_trn,num_tst,num_val);

  opts.objective = 'msvm_smooth';
  opts.epsilon = 1e-10;
  opts.max_epoch = 100;
  opts.eval_epoch = 5;
  % opts.log_level = 'verbose';

  model = matsdca_fit({Xtrn, Xval, Xtst}, {Ytrn, Yval, Ytst}, opts);
  disp(model);

  % Further details:
  % matsdca_fit('help')

end
