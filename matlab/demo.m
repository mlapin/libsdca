%
% Demo script for testing matsdca mex files
%
clear;
rng(0);

% Test prox
if 1
  fprintf('Demo: prox\n');
  d = 100;
  n = 5;

  A = randn(d,n);

  opts = [];
  opts.prox = 'knapsack';
  B = matsdca_prox(A, opts);

  fprintf('All columns should sum up to 1.\n');
  disp(sum(B));

  % Further details:
  % matsdca_prox('help')

end


% Test solver
if 1
  fprintf('Demo: solver\n');
  num_trn = 200;
  num_val = 200;
  num_tst = 200000;

  [Xtrn,Ytrn,Xval,Yval,Xtst,Ytst] = getdata(num_trn,num_val,num_tst);

  opts = [];
  opts.objective = 'softmax'; % or 'msvm_smooth'
  opts.c = 0.1;
  opts.epsilon = 1e-3;
  opts.max_epoch = 100;
  opts.eval_epoch = 1; % only for demo, in practice would be wasteful
%   opts.log_level = 'verbose'; % or 'none'

  model = matsdca_fit({Xtrn, Xval, Xtst}, {Ytrn, Yval, Ytst}, opts);
  disp(model);
  fprintf('trn accuracy: %.2f\n', 100*model.train(end).accuracy);
  fprintf('val accuracy: %.2f\n', 100*model.test(end,1).accuracy);
  fprintf('tst accuracy: %.2f\n', 100*model.test(end,2).accuracy);

  % Further details:
  % matsdca_fit('help')

end
