function [Xtrn,Ytrn,Xval,Yval,Xtst,Ytst] = getdata(num_trn,num_val,num_tst)
% Generate data for the synthetic example presented in
% Lapin, M., Hein, M. and Schiele, B.
% Loss Functions for Top-k Error: Analysis and Insights,
% In CVPR 2016.
%

num_examples = num_trn + num_val + num_tst;

seg_length = [1, 1, 1, 3, 1];
seg_class = [...
  0, 1, .4, .3, 0; ... % class 1
  1, 0, .1, .7, 0; ... % class 2
  0, 0, .5,  0, 1; ... % class 3
  ];
num_segments = numel(seg_length);

seg_offsets = [0, cumsum(seg_length)];
seg_cdf = cumsum(seg_class);

% Generate examples
rng(0);
seg_id = randi(num_segments, num_examples, 1);
X = rand(1, num_examples);
Y = rand(1, num_examples);
for i = 1:num_examples
  X(i) = X(i) * seg_length(seg_id(i)) + seg_offsets(seg_id(i));
  Y(i) = find(Y(i) <= seg_cdf(:,seg_id(i)),1);
end

% Map onto the unit cirlce in 2D
t = 2*pi*X/max(seg_offsets);
X = [cos(t); sin(t)];

% Train / validation / test splits
Xtrn = X(:,1:num_trn);
Ytrn = Y(1:num_trn);
Xval = X(:,num_trn+1:num_trn+num_val);
Yval = Y(num_trn+1:num_trn+num_val);
Xtst = X(:,num_trn+num_val+1:end);
Ytst = Y(num_trn+num_val+1:end);

if 0
fprintf('Class probabilities:\n');
fprintf('  %.3f\n', seg_class * ones(num_segments,1) / num_segments);

num_classes = size(seg_class,1);
num_per_class = hist(Y, 1:num_classes);
fprintf('Empirical class probabilities:\n');
fprintf('  %.3f\n', num_per_class / num_examples);
end
