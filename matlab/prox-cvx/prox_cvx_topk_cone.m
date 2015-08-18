function [X,mu,nu] = prox_cvx_topk_cone(A, opts)
%#ok<*ASGLU>
%#ok<*EQEFF>
%#ok<*STOUT>
%#ok<*VUNUS>

k = opts.k;

[d,n] = size(A);

cvx_begin quiet
  cvx_precision high;
  variable X(d,n);
  dual variable mu;
  dual variable nu;
  minimize( 0.5*sum(sum((A - X).^2)) );
  mu: 0 <= X;
  nu: X <= ones(d)*X/k;
cvx_end
