%
% Create data to test prox operators.
%

rng(1);
version = 1;
fname = fullfile('data', sprintf('prox_test_data-v%d.mat', version));

data = {}; %#ok<*SAGROW>

% d = 1
data{end+1} = 0;
data{end+1} = 1;
data{end+1} = -1;
data{end+1} = eps;
data{end+1} = 1/eps;
data{end+1} = double(eps('single'));
data{end+1} = double(1/eps('single'));

% Id
data{end+1} = eye(2);
data{end+1} = eye(5);
data{end+1} = eye(10);
data{end+1} = eye(100);
data{end+1} = -eye(100);

% Scaled all ones vector
for d = [2 5 10 100 1000 10000]
  data{end+1} = zeros(d,1);
  for x = [0 1 10 80]
    data{end+1} = exp(x)*ones(d,1);
    data{end+1} = -exp(x)*ones(d,1);
    if x > 0
      data{end+1} = exp(-x)*ones(d,1);
      data{end+1} = -exp(-x)*ones(d,1);
    end
  end
end

N = 100;

% Random -1,0,1
for d = [2 5 10 100 1000]
  x = 3*rand(d,N);
  x(x<1) = -1;
  x(1<=x & x<=2) = 0;
  x(2<x) = 1;
  data{end+1} = x;
  clear x;
end

% Uniform
for d = [10 100 1000]
  for a = [-10 0 10 100]
    for w = [1e-5 1 100 1e5]
      data{end+1} = a + w*rand(d,N);
    end
  end
end

% Gaussian
for d = [10 100 1000]
  for a = [-10 0 10 100]
    for w = [1e-5 1 100 1e5]
      data{end+1} = a + w*randn(d,N);
    end
  end
end

% Exponential
for d = [100 1000]
  for a = [0 100]
    for w = [1e-3 1 1e3]
      data{end+1} = a + exprnd(w,d,N);
    end
  end
end

% Generalized extreme value
for d = [100 500]
  for a = [-100 1 100]
    for w = [1e-2 1 1e2]
      for s = [-1 0 1]
        data{end+1} = gevrnd(s,w,a,d,N);
      end
    end
  end
end

data = data';
whos('data')

save(fname, 'data', 'version');
fprintf('Test data saved to:\n%s\n', fname);
