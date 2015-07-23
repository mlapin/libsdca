clear;
close all;
addpath('libsdca-release');

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

  opts.objective = 'l2_hinge_topk';
  opts.k = 10;
  opts.gamma = 0;
  opts.epsilon = 1e-5;
  opts.check_epoch = 5;
  opts.max_num_epoch = 100;
  opts.precision = 'double';
  opts.log_level = 'debug';
  opts.log_format = 'long_e';
  opts.is_dual = 0;

  if opts.is_dual
    model = libsdca_solve(single(Xtrn)'*single(Xtrn), Ytrn, opts);
    disp(model);
    [~,pred] = max(model.A*Xtrn'*Xtrn);
    fprintf('accuracy: %g\n', 100*mean(pred(:) == Ytrn(:)));
  else
    model = libsdca_solve(single(Xtrn), Ytrn, opts);
    disp(model);
    [~,pred] = max(model.W'*Xtrn);
    fprintf('accuracy: %g\n', 100*mean(pred(:) == Ytrn(:)));
  end
  
  if 0
  opts2 = model;
  opts2.check_on_start = true;
  opts2.k = 10;
  
  opts_prox.proj = 'topk_simplex_biased';
  opts_prox.k = opts2.k;
  opts_prox.rhs = opts2.C;
  opts_prox.rho = 1;
  libsdca_prox(opts2.A, opts_prox);
  
  model1 = libsdca_solve(single(Xtrn), Ytrn, opts2);
  disp(model1);
  [~,pred] = max(model1.W'*Xtrn);
  fprintf('accuracy: %g\n', 100*mean(pred(:) == Ytrn(:)));
  
  opts3 = model1;
  opts3.summation = 'kahan';
  opts3.precision = 'long double';
  model2 = libsdca_solve(single(Xtrn), Ytrn, opts3);
  disp(model2);
  [~,pred] = max(model2.W'*Xtrn);
  fprintf('accuracy: %g\n', 100*mean(pred(:) == Ytrn(:)));
  end
end
