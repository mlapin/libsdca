function [X,mu,nu,lambda] = prox_cvx_knapsack_le(A, opts)
%#ok<*ASGLU>
%#ok<*EQEFF>
%#ok<*STOUT>
%#ok<*VUNUS>

lo = opts.lo;
hi = opts.hi;
rhs = opts.rhs;

[d,n] = size(A);

cvx_begin quiet
  cvx_precision high;
  variable X(d,n);
  dual variable mu;
  dual variable nu;
  dual variable lambda;
  minimize( 0.5*sum(sum((A - X).^2)) );
  mu: lo <= X;
  nu: X <= hi;
  lambda: sum(X,1) <= rhs;
cvx_end
