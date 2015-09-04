function [accuracy] = allaccuracies(scores, labels, topK)
%ALLACCURACIES Compute top-1, top-2, ..., top-K accuracies
%
%  ACCURACY = ALLACCURACIES(SCORES, LABELS) Compute all top-K accuracies
%  ACCURACY = ALLACCURACIES(SCORES, LABELS, TOPK) Compute top-K accuracies
%
% Inputs:
%  SCORES   - C-by-N matrix of classifier scores,
%             where C is the number of classes and N is the sample size
%  LABELS   - 1-by-N vector of ground truth labels,
%             where each label is in 1:C
%  TOPK     - M-dim vector of the required K's in the top-K measure
%             or
%             scalar of the largest required K (identical to 1:TOPK)
%             or
%             None, defaults to topK = 1:C
%
% Outputs:
%  ACCURACY - M-by-1 vector of top-K accuracies
%

% Copyright (C) 2014 Maksim Lapin.

narginchk(2, 3);
if nargin < 3
  topK = 1:size(scores, 1);
elseif numel(topK) == 1
  topK = 1:min(size(scores, 1), topK);
end

K = max(topK);
[~,IX] = sort(scores, 1, 'descend');
S = cumsum(bsxfun(@eq, IX(1:K, :), labels));
accuracy = mean(S,2);
