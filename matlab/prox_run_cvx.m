% function prox_run_cvx(prox)
% Run CVX solver PROX on test data and save results
%

run(fullfile('cvx', 'cvx_startup'));
addpath('prox-cvx');

%#ok<*SAGROW>

% if nargin < 1, prox = 'all'; end
prox = 'all';

version = 1;
fdata = fullfile('data', sprintf('prox_test_data-v%d.mat', version));
data = load(fdata, 'data');
data = data.data;

batch = 10;
num_data = numel(data);

k_vals = [1 3 5 10 20 50];
rhs_vals = 10.^(-6:2:6);

if strcmp(prox, 'prox_cvx_entropy') || strcmp(prox, 'all')
  fprintf('prox_cvx_entropy\n');
  fres = fullfile('data', sprintf('prox_cvx_entropy-v%d.mat', version));
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
    fprintf('i = %d (%d)\n', i, num_data);
  end
  save(fres, 'results', 'version');
end
