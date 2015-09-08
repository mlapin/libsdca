% clear;
% close all;
addpath('libsdca-debug');
% addpath('libsdca-release');
rng(0);

if usejava('jvm') && ~exist('cvx_begin', 'file') ...
    && exist(fullfile('cvx', 'cvx_startup.m'), 'file')
  addpath('prox-cvx');
  run(fullfile('cvx', 'cvx_startup.m'));
  % SeDuMi is usually faster, but SDPT3 may be more accurate
  cvx_solver sedumi
%   cvx_solver sdpt3;
end



if 0
  d = 10;
  n = 10;

%   opts.prox = 'entropy';
%   opts.prox = 'topk_cone_biased';
%   opts.prox = 'knapsack';
%   opts.prox = 'lambert_w_exp';
  opts.prox = 'topk_entropy';
%   opts.prox = 'topk_entropy_biased';
%   opts.prox = 'topk_simplex_biased';
  opts.k = 10;
%   opts.alpha = 1e+3;
%   opts.summation = 'kahan';
%   opts.rhs = 1;
%   opts.hi = 1;

%   A = bsxfun(@times, ones(d,n), randn(1,n));
  A = randn(d,n) + 10*randn(d,n) + 100*randn(d,n) + 1000*randn(d,n);
%   A = -10:0.01:10;
  B = libsdca_prox(A, opts);
  
if exist('cvx_begin', 'file')
  [X,info] = prox_cvx(A, opts);

% [X,mu,nu] = prox_cvx_entropy(A, opts);
% loss = @(A,X) 0.5*sum(sum((A - X).^2)) - sum(sum(entr(X)));
  
  disp(opts);
  loss = info.loss;
  fprintf('Loss (lower is better):\n');
  fprintf('      lib = %+.16e\n', loss(A,B));
  fprintf('      cvx = %+.16e\n', loss(A,X));
  fprintf('cvx - lib = %+.16e\n', loss(A,X) - loss(A,B));
  fprintf('Solution difference:\n');
  fprintf('     RMSD = %+.16e\n', norm(B-X,'fro')/sqrt(numel(B)));
  sum(B)
  sum(X)
  k=opts.k;
  
end  
%   T = zeros(100,1);
%   for k=1:100
%     opts.k = k;
%     t = tic;
%     B = libsdca_prox(A, opts);
%     T(k) = toc(t);
%   end
%     plot(T)
%   plot(sum(B))
%   disp(sum(B));
  
%   [X,mu,nu] = prox_entropy_cvx(A, opts.hi, opts.rhs);
%   
%   loss = @(X) 0.5*sum(sum((A - X).^2)) - sum(sum(entr(X)));
%   disp(loss(X)-loss(B));
end

if 0
  cd /BS/mlapin-projects1/work/simplex/test
  runtestcases_2
end

if 1
%   load('data/sun397-cnn.mat');
  load('data/sun397-fv-trn.mat');
  load('data/sun397-fv-tst.mat');
%   Ktrn = Ktrn-1;
  
%   ix = 1:5*2;
%   Ktrn = Ktrn(ix,ix);
%   Ytrn = Ytrn(ix);
  

  opts.objective = 'l2_entropy_topk';
%   opts.objective = 'l2_topk_hinge';
%   opts.objective = 'l2_hinge_topk';
  opts.C = 1;
  opts.k = 2;
  opts.gamma = 0;
  opts.epsilon = 1e-15;
  opts.check_on_start = 0;
  opts.check_epoch = 1;
  opts.max_epoch = 50;
  opts.summation = 'standard';
  opts.precision = 'double';
  opts.log_level = 'debug';
  opts.log_format = 'long_e';
  opts.is_dual = 1;

  if opts.is_dual
    if ~exist('Ktrn', 'var')
      Ktrn = Xtrn'*Xtrn;
    end
    model = libsdca_solve({Ktrn, Ktst}, {Ytrn, Ytst}, opts);
    disp(model);
    [~,pred] = max(model.A*Ktrn);
    fprintf('accuracy: %g\n', 100*mean(pred(:) == Ytrn(:)));
  else
    model = libsdca_solve(Xtrn, Ytrn, opts);
    disp(model);
    [~,pred] = max(model.W'*Xtrn);
    fprintf('accuracy: %g\n', 100*mean(pred(:) == Ytrn(:)));
  end
  disp(model.records);
  disp(model.evals);
  size([model.records.epoch])
  size([model.evals.loss])
  size([model.evals.accuracy])
  
  if 0
    opts2 = model;
    opts2.gamma = 0;
    opts2.check_on_start = true;
    if opts2.is_dual
      model2 = libsdca_solve(Xtrn'*Xtrn, Ytrn, opts2);
      disp(model2);
      [~,pred] = max(model2.A*Xtrn'*Xtrn);
      fprintf('accuracy: %g\n', 100*mean(pred(:) == Ytrn(:)));
    else
      model2 = libsdca_solve(Xtrn, Ytrn, opts2);
      disp(model2);
      [~,pred] = max(model2.W'*Xtrn);
      fprintf('accuracy: %g\n', 100*mean(pred(:) == Ytrn(:)));
    end
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
