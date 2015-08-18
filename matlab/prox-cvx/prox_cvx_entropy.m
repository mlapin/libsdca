function [X,mu,nu] = prox_cvx_entropy(A, opts)
%#ok<*ASGLU>
%#ok<*EQEFF>
%#ok<*STOUT>
%#ok<*VUNUS>

hi = opts.hi;
rhs = opts.rhs;

[d,n] = size(A);

cvx_begin quiet
  cvx_expert true;
  cvx_precision high;
  variable X(d,n);
  dual variable mu;
  dual variable nu;
  dual variable lambda;
  minimize( 0.5*sum(sum((A - X).^2)) - sum(sum(entr(X))) );
  mu: 0 <= X;
  nu: X <= hi;
  lambda: sum(X,1) == rhs;
cvx_end
