%
% Run CVX solver PROX on test data and save results
%

run(fullfile('cvx', 'cvx_startup'));
addpath('prox-cvx');

%#ok<*SAGROW>

version = 1;
fdata = fullfile('data', sprintf('prox_test_data-v%d.mat', version));
data = load(fdata, 'data');
data = data.data;

whos('data')

batch = 10;
num_data = numel(data);

lo_vals = [-1 0 1];
k_vals = [1 3 5 10 20];
rhs_vals = 10.^(-5:2:5);
rho_vals = [0 1 1e-3 1e+3];


tStart = tic;

if strcmp(prox, 'prox_cvx_entropy')
  fprintf('%s\n', prox);
  fres = fullfile('data', sprintf('%s-v%d.mat', prox, version));
  results = cell(num_data,1);
  parfor i = 1:num_data
    opts = [];
    opt = struct('i', i);
    A = data{i};
    for rhs = rhs_vals
      opt.rhs = rhs;
      for k = k_vals
        if k > size(A,1), continue; end

        opt.hi = rhs/k;
        X = nan(size(A));
        for i1 = 1:batch:size(A,2)
          i2 = min(size(A,2), i1+batch);
          X(:,i1:i2) = prox_cvx_entropy(A(:,i1:i2), opt);
        end
        opt.X = X;
        
        if isempty(opts), opts = opt; else opts(end+1) = opt; end
      end
    end
    results{i} = opts;
    fprintf('[%8.2f] i = %d (%d)\n', toc(tStart), i, num_data);
  end
  save(fres, 'results', 'version');
end

if strcmp(prox, 'prox_cvx_knapsack_eq')
  fprintf('%s\n', prox);
  fres = fullfile('data', sprintf('%s-v%d.mat', prox, version));
  results = cell(num_data,1);
  parfor i = 1:num_data
    opts = [];
    opt = struct('i', i);
    A = data{i};
    for rhs = rhs_vals
      opt.rhs = rhs;
      for k = k_vals
        if k > size(A,1), continue; end

        opt.hi = rhs/k;
        for lo = lo_vals
          if lo > opt.hi, continue; end
          
          opt.lo = lo;
          X = nan(size(A));
          for i1 = 1:batch:size(A,2)
            i2 = min(size(A,2), i1+batch);
            X(:,i1:i2) = prox_cvx_knapsack_eq(A(:,i1:i2), opt);
          end
          opt.X = X;

          if isempty(opts), opts = opt; else opts(end+1) = opt; end
        end
      end
    end
    results{i} = opts;
    fprintf('[%8.2f] i = %d (%d)\n', toc(tStart), i, num_data);
  end
  save(fres, 'results', 'version');
end

if strcmp(prox, 'prox_cvx_knapsack_le')
  fprintf('%s\n', prox);
  fres = fullfile('data', sprintf('%s-v%d.mat', prox, version));
  results = cell(num_data,1);
  parfor i = 1:num_data
    opts = [];
    opt = struct('i', i);
    A = data{i};
    for rhs = rhs_vals
      opt.rhs = rhs;
      for k = k_vals
        if k > size(A,1), continue; end

        opt.hi = rhs/k;
        for lo = lo_vals
          if lo > opt.hi, continue; end
          
          opt.lo = lo;
          X = nan(size(A));
          for i1 = 1:batch:size(A,2)
            i2 = min(size(A,2), i1+batch);
            X(:,i1:i2) = prox_cvx_knapsack_le(A(:,i1:i2), opt);
          end
          opt.X = X;

          if isempty(opts), opts = opt; else opts(end+1) = opt; end
        end
      end
    end
    results{i} = opts;
    fprintf('[%8.2f] i = %d (%d)\n', toc(tStart), i, num_data);
  end
  save(fres, 'results', 'version');
end

if strcmp(prox, 'prox_cvx_knapsack_le_biased')
  fprintf('%s\n', prox);
  fres = fullfile('data', sprintf('%s-v%d.mat', prox, version));
  results = cell(num_data,1);
  parfor i = 1:num_data
    opts = [];
    opt = struct('i', i);
    A = data{i};
    for rhs = rhs_vals
      opt.rhs = rhs;
      for k = k_vals
        if k > size(A,1), continue; end

        opt.hi = rhs/k;
        for lo = lo_vals
          if lo > opt.hi, continue; end
          
          opt.lo = lo;
          for rho = rho_vals
            opt.rho = rho;
            X = nan(size(A));
            for i1 = 1:batch:size(A,2)
              i2 = min(size(A,2), i1+batch);
              X(:,i1:i2) = prox_cvx_knapsack_le_biased(A(:,i1:i2), opt);
            end
            opt.X = X;

            if isempty(opts), opts = opt; else opts(end+1) = opt; end
          end
        end
      end
    end
    results{i} = opts;
    fprintf('[%8.2f] i = %d (%d)\n', toc(tStart), i, num_data);
  end
  save(fres, 'results', 'version');
end

if strcmp(prox, 'prox_cvx_topk_cone')
  fprintf('%s\n', prox);
  fres = fullfile('data', sprintf('%s-v%d.mat', prox, version));
  results = cell(num_data,1);
  parfor i = 1:num_data
    opts = [];
    opt = struct('i', i);
    A = data{i};
    for k = k_vals
      opt.k = k;
      X = nan(size(A));
      for i1 = 1:batch:size(A,2)
        i2 = min(size(A,2), i1+batch);
        X(:,i1:i2) = prox_cvx_topk_cone(A(:,i1:i2), opt);
      end
      opt.X = X;

      if isempty(opts), opts = opt; else opts(end+1) = opt; end
    end
    results{i} = opts;
    fprintf('[%8.2f] i = %d (%d)\n', toc(tStart), i, num_data);
  end
  save(fres, 'results', 'version');
end

if strcmp(prox, 'prox_cvx_topk_cone_biased')
  fprintf('%s\n', prox);
  fres = fullfile('data', sprintf('%s-v%d.mat', prox, version));
  results = cell(num_data,1);
  parfor i = 1:num_data
    opts = [];
    opt = struct('i', i);
    A = data{i};
    for k = k_vals
      opt.k = k;
      for rho = rho_vals
        opt.rho = rho;
        X = nan(size(A));
        for i1 = 1:batch:size(A,2)
          i2 = min(size(A,2), i1+batch);
          X(:,i1:i2) = prox_cvx_topk_cone_biased(A(:,i1:i2), opt);
        end
        opt.X = X;

        if isempty(opts), opts = opt; else opts(end+1) = opt; end
      end
    end
    results{i} = opts;
    fprintf('[%8.2f] i = %d (%d)\n', toc(tStart), i, num_data);
  end
  save(fres, 'results', 'version');
end

if strcmp(prox, 'prox_cvx_topk_simplex')
  fprintf('%s\n', prox);
  fres = fullfile('data', sprintf('%s-v%d.mat', prox, version));
  results = cell(num_data,1);
  parfor i = 1:num_data
    opts = [];
    opt = struct('i', i);
    A = data{i};
    for k = k_vals
      opt.k = k;
      for rhs = rhs_vals
        opt.rhs = rhs;
        X = nan(size(A));
        for i1 = 1:batch:size(A,2)
          i2 = min(size(A,2), i1+batch);
          X(:,i1:i2) = prox_cvx_topk_simplex(A(:,i1:i2), opt);
        end
        opt.X = X;

        if isempty(opts), opts = opt; else opts(end+1) = opt; end
      end
    end
    results{i} = opts;
    fprintf('[%8.2f] i = %d (%d)\n', toc(tStart), i, num_data);
  end
  save(fres, 'results', 'version');
end

if strcmp(prox, 'prox_cvx_topk_simplex_biased')
  fprintf('%s\n', prox);
  fres = fullfile('data', sprintf('%s-v%d.mat', prox, version));
  results = cell(num_data,1);
  parfor i = 1:num_data
    opts = [];
    opt = struct('i', i);
    A = data{i};
    for k = k_vals
      opt.k = k;
      for rhs = rhs_vals
        opt.rhs = rhs;
        for rho = rho_vals
          opt.rho = rho;
          X = nan(size(A));
          for i1 = 1:batch:size(A,2)
            i2 = min(size(A,2), i1+batch);
            X(:,i1:i2) = prox_cvx_topk_simplex_biased(A(:,i1:i2), opt);
          end
          opt.X = X;

          if isempty(opts), opts = opt; else opts(end+1) = opt; end
        end
      end
    end
    results{i} = opts;
    fprintf('[%8.2f] i = %d (%d)\n', toc(tStart), i, num_data);
  end
  save(fres, 'results', 'version');
end
