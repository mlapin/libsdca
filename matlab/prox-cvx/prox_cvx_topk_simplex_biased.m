function [X,lambda,nu,mu] = prox_cvx_topk_simplex_biased(A, opts)
%#ok<*ASGLU>
%#ok<*EQEFF>
%#ok<*STOUT>
%#ok<*VUNUS>

k = opts.k;
rhs = opts.rhs;
rho = opts.rho;

[d,n] = size(A);

cvx_begin quiet
  cvx_precision high;
  variable X(d,n);
  dual variable mu;
  dual variable nu;
  dual variable lambda;
  minimize( 0.5*sum(sum((A - X).^2)) + 0.5*rho*sum(sum(X,1).^2) );
  mu: 0 <= X;
  nu: X <= ones(d)*X/k;
  lambda: sum(X) <= rhs;
cvx_end
