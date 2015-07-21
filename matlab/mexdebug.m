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

if 0
  cd /BS/mlapin-projects1/work/simplex/test
  runtestcases_2
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

  opts.epsilon = 1e-15;
  opts.check_epoch = 10;
  opts.max_num_epoch = 1000;
  opts.log_level = 'debug';
  opts.log_format = 'long_e';

  model = libsdca_solve(Xtrn, Ytrn, opts);
  disp(model);
  [~,pred] = max(model.W'*Xtrn);
  fprintf('accuracy: %g\n', 100*mean(pred(:) == Ytrn(:)));
end
