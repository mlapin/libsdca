function [X,loss,lambda,nu,mu] = prox_cvx_topk_entropy(A, opts)
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
  dual variable mu;
  dual variable nu;
  dual variable lambda;
  minimize( - A(:)'*X(:) - sum(entr(X(:))) - sum(entr(1 - sum(X,1))) );
  mu: 0 <= X;
  nu: X <= ones(d)*X/k;
  lambda: sum(X,1) <= 1;
cvx_end

loss = @(A,X) -A(:)'*X(:) - sum(entr(max(0,X(:)))) ...
  - sum(entr( 1 - min(1,sum(X,1)) ));
  