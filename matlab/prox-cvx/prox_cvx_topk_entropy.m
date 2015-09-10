function [X,info] = prox_cvx_topk_entropy(A, opts)
%#ok<*ASGLU>
%#ok<*EQEFF>
%#ok<*STOUT>
%#ok<*VUNUS>
%#ok<*NODEF>

k = opts.k;

[d,n] = size(A);

cvx_begin %quiet
  cvx_expert true;
  cvx_precision high;
  variable X(d,n);
  variable s(1,n);
  dual variable t;
  dual variable mu;
  dual variable nu;
  dual variable lambda;
  minimize( - A(:)'*X(:) - sum(entr(X(:))) - sum(entr(1 - s)) );
  t: sum(X,1) == s;
  mu: 0 <= X;
  nu: X <= ones(d,1) * s/k;
  lambda: s <= 1;
cvx_end

info.loss = @(A,X) - A(:)'*max(0,X(:)) - sum(entr(max(0,X(:)))) ...
  - sum(entr(1 - min(1,sum(max(0,X),1)) ));

info.k = k;
info.s = s;
info.t = t;
info.mu = mu;
info.nu = nu;
info.lambda = lambda;
