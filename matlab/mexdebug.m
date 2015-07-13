clear;
close all;
addpath('libsdca');

if 0
  d = 10;
  n = 5;

  opts.proj = 'topk_simplex';
  opts.rhs = .75;
  opts.rho = 1.5;
  opts.k = 3;

  A = randn(d,n);
  B = libsdca_prox(A, opts);
end

if 1
  load('data/sun397-cnn.mat');
  top_k = 10;
  svm_c = 10;
  gamma = 0;
  epsilon = 0.001;
  check_gap_frequency = 1;
  max_num_epoch = 100;
  max_wall_time = 0;
  max_cpu_time = 0;

  model = libsdca_solve(Xtrn, Ytrn);
  disp(model);
  [~,pred] = max(model.W'*Xtrn);
  fprintf('accuracy: %g\n', 100*mean(pred(:) == Ytrn(:)));
end