clc;
clear;
close all;
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

%%%
%%% Test prox on real data
%%%
if 1
  load('data/prox_topk_simplex_biased.mat');
  ix = 1:20;
  A = A(:,ix);
  X1 = libsdca_prox(A, opts);
  [X2,info] = prox_cvx(A, opts);

  disp(opts);
  loss = info.loss;
  fprintf('Loss (lower is better):\n');
  fprintf('   solver = %+.16e\n', loss(A,X(:,ix)));
  fprintf('      lib = %+.16e\n', loss(A,X1));
  fprintf('      cvx = %+.16e\n', loss(A,X2));
  fprintf('cvx - lib = %+.16e\n', loss(A,X2) - loss(A,X1));
  fprintf('Solution difference:\n');
  fprintf('     RMSD = %+.16e\n', norm(X1-X2,'fro')/sqrt(numel(X1)));
end

%%%
%%% Test prox
%%%
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

%%%
%%% Run test cases
%%%
if 0
  cd /BS/mlapin-projects1/work/simplex/test
  runtestcases_2
end

%%%
%%% Test solver
%%%
if 0
%   load('data/sun397-cnn.mat');
%   load('data/sun397-cnn-trn.mat');
%   load('data/sun397-cnn-tst.mat');
%   load('data/sun397-fv.mat'); % converges
%   load('data/sun397-fv-trn.mat');
%   load('data/sun397-fv-tst.mat');
  load('data/indoor67-cnn-trn.mat'); % no convergence
%   load('data/indoor67-cnn-tst.mat');
% 
%   Xc = mean(Xtrn,2);
%   Xtrn = bsxfun(@minus, Xtrn, Xc);
%   Xtst = bsxfun(@minus, Xtst, Xc);
%   Xtrn = [Xtrn; ones(1, size(Xtrn,2))];
%   Xtst = [Xtst; ones(1, size(Xtst,2))];
  Xtrn = double(Xtrn);
%   Xtst = double(Xtst);

%  Ktrn = double(Ktrn);
%   Ktrn = Ktrn-1;
  
%   ix = 1:5*2;
%   Ktrn = Ktrn(ix,ix);
%   Ytrn = Ytrn(ix);

%   opts.objective = 'l2_entropy_topk';
%   opts.objective = 'l2_topk_hinge';
  opts.objective = 'l2_hinge_topk';
  opts.C = 1;
  opts.k = 1;
  opts.gamma = 0;
  opts.epsilon = 1e-15;
  opts.check_on_start = 0;
  opts.check_epoch = 10;
  opts.max_epoch = 150;
  opts.summation = 'standard';
  opts.precision = 'double';
  opts.log_level = 'debug';
  opts.log_format = 'long_e';
  opts.is_dual = 1;

  if opts.is_dual
    if ~exist('Ktrn', 'var') && exist('Xtrn', 'var')
      Ktrn = Xtrn'*Xtrn;
    end
    if ~exist('Ktst', 'var') && exist('Xtst', 'var')
      Ktst = Xtrn'*Xtst;
    end
    if exist('Ktst', 'var')
      model = libsdca_solve({Ktrn, Ktst}, {Ytrn, Ytst}, opts);
    else
      model = libsdca_solve(Ktrn, Ytrn, opts);
    end
    disp(model);
    [~,pred] = max(model.A*Ktrn);
    fprintf('accuracy: %g\n', 100*mean(pred(:) == Ytrn(:)));
  else
    if exist('Xtst', 'var')
      model = libsdca_solve({Xtrn, Xtst}, {Ytrn, Ytst}, opts);
    else
      model = libsdca_solve(Xtrn, Ytrn, opts);
    end
    disp(model);
    [~,pred] = max(model.W'*Xtrn);
    fprintf('accuracy: %g\n', 100*mean(pred(:) == Ytrn(:)));
  end
  
end
