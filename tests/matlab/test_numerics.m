function test_numerics
% Run after make.m. Scores are classes-by-examples; labels are 1-based.
z = [20; -20; -20];
[v,g] = matsdca_loss(z,1);
assert(v > 0 && g(1) < 0);
assert(abs(sum(g)) < 1e-30);
v = matsdca_loss(ones(3,1)*1e16,1);
assert(abs(v-log(3)) < 1e-14);
% Central differences match the gradients; each loss takes only its own parameters.
z = [.3; -.8; 1.2; .1];
cases = {'cross_entropy', struct(); 'topk_cross_entropy', struct('k',2); ...
  'topk_hinge_alpha', struct('k',2,'gamma',.7); 'topk_hinge_beta', struct('k',2,'gamma',.7); ...
  'multiclass_hinge', struct('gamma',.7); 'multilabel_hinge', struct('gamma',.7); ...
  'multilabel_cross_entropy', struct(); 'proximal_cross_entropy', struct('alpha',1.2); ...
  'distillation', struct('temperature',2); ...
  'proximal_distillation', struct('temperature',2,'alpha',.5)};
for i=1:size(cases,1)
  opts = cases{i,2}; opts.loss = cases{i,1};
  target = 1;
  if strncmp(cases{i,1},'multilabel',10), target=logical([1;1;0;0]); end
  if ~isempty(strfind(cases{i,1},'distillation')), target=[0;1;-1;2]; end
  [~,g] = matsdca_loss(z,target,opts);
  for j=1:numel(z)
    plus=z; minus=z; plus(j)=plus(j)+1e-5; minus(j)=minus(j)-1e-5;
    numeric=(matsdca_loss(plus,target,opts)-matsdca_loss(minus,target,opts))/2e-5;
    assert(abs(numeric-g(j)) < 1e-6, cases{i,1});
  end
end
% A tiny smoothing parameter still enforces the total projection mass.
for gamma=[1e-10,1e-20,1e-100]
  for k=1:3
    [v,g]=matsdca_loss(zeros(4,1),1, ...
      struct('loss','topk_hinge_beta','k',k,'gamma',gamma));
    assert(abs(v-(1-gamma/6)) < 2e-15);
    assert(max(abs(g-[-1;1/3;1/3;1/3])) < 2e-15);
  end
end
% Equal student and teacher logits give an exact zero distillation gradient.
[~,g]=matsdca_loss(z,z,struct('loss','distillation','temperature',4));
assert(all(g == 0));
q=matsdca_prox(-20,struct('prox','topk_entropy_biased','alpha',1));
assert(q > 0);
assert_error(@() matsdca_loss(z,1,struct('loss','topk_hinge_beta','k',1.5)));
assert_error(@() matsdca_loss(z,0));
assert_error(@() matsdca_prox(z,struct('k',-1)));
% Every solver dispatch on a small problem.
X=[1 0 -1 .5; 0 1 .5 -1]; y=[1 2 3 1];
Y=sparse(y,1:numel(y),ones(size(y)),3,numel(y));
fits={'cross_entropy',struct(); 'topk_cross_entropy',struct('k',2); ...
  'multiclass_hinge',struct(); 'multiclass_hinge',struct('gamma',1); ...
  'topk_hinge_alpha',struct('k',2); 'topk_hinge_beta',struct('k',2,'gamma',.5); ...
  'multilabel_hinge',struct(); 'multilabel_hinge',struct('gamma',1); ...
  'multilabel_cross_entropy',struct()};
for i=1:size(fits,1)
  opts=fits{i,2}; opts.loss=fits{i,1}; opts.max_epoch=1000;
  target=y;
  if strncmp(fits{i,1},'multilabel',10), target=Y; end
  model=matsdca_fit(X,target,opts);
  assert(isstruct(model) && strcmp(model.loss,fits{i,1}));
end
test_names(X,y);
test_k(X,y);
test_prox_precision();
test_warm_start();
test_sparse_targets(X);
test_ordinary_inputs();
disp('libsdca MATLAB numerical smoke tests passed');
end

function test_k(X,y)
% The top-k losses need 1 <= k < classes (3 classes here); the others take no k.
Y=sparse(y,1:numel(y),1,3,numel(y));
z=[.3 -1 .2 .1; -.8 .4 .5 -.1; 1.2 .2 -.3 .4];
names={'cross_entropy','topk_cross_entropy','multiclass_hinge','topk_hinge_alpha', ...
  'topk_hinge_beta','multilabel_hinge','multilabel_cross_entropy','proximal_cross_entropy'};
for i=1:numel(names)
  multilabel=strncmp(names{i},'multilabel',10);
  topk=~isempty(strfind(names{i},'topk'));
  target=y; scores=y;
  if multilabel, target=Y; scores=full(Y)>0; end
  fit=~strcmp(names{i},'proximal_cross_entropy');
  opts=struct('loss',names{i});
  if topk
    for k=[0 3]
      assert_error(@() matsdca_loss(z,scores,setfield(opts,'k',k)));
      if fit, assert_error(@() matsdca_fit(X,target,setfield(opts,'k',k))); end
    end
    assert_error(@() matsdca_loss(z,scores,opts));  % k is required
    opts.k=2;
  else
    assert_error(@() matsdca_loss(z,scores,setfield(opts,'k',1)));
    if fit, assert_error(@() matsdca_fit(X,target,setfield(opts,'k',1))); end
  end
  assert(all(isfinite(matsdca_loss(z,scores,opts))));
  if fit
    assert(isstruct(matsdca_fit(X,target,opts)));
  end
end
end

function test_names(X,y)
% Shared names only: the v0.2 names and option aliases are errors, as are unknown options.
model=matsdca_fit(X,y);
assert(strcmp(model.loss,'cross_entropy') && strcmp(model.precision,'float64'));
for name={'cross_entropy','topk_hinge_alpha','topk_hinge_beta'}
  opts=struct('loss',name{1});
  if ~strcmp(name{1},'cross_entropy'), opts.k=2; end
  model=matsdca_fit(X,y,opts);
  assert(strcmp(model.loss,name{1}));
end
for name={'softmax','msvm','msvm_smooth','mlsoftmax','mlsvm','l2_topk_hinge','topk_hinge'}
  assert_error(@() matsdca_fit(X,y,struct('loss',name{1})));
end
for alias={'c','epsilon','is_dual','W0','A','W','objective','max_cpu_time','max_wall_time'}
  assert_error(@() matsdca_fit(X,y,struct(alias{1},1)));
end
assert_error(@() matsdca_fit(X,y,struct('gamma',1)));      % cross_entropy takes no gamma
assert_error(@() matsdca_fit(X,y,struct('shrinking','off')));  % nor shrinking
assert_error(@() matsdca_fit(X,y,struct('loss','multiclass_hinge','projection_precision','float64')));
z=[.3 -1; -.8 .4; 1.2 .2; .1 .5]; mask=logical([1 0; 1 1; 0 0; 0 0]);
% One target kind per loss: cross_entropy takes labels, the multilabel losses label sets.
assert_error(@() matsdca_loss(z,mask,struct('loss','cross_entropy')));
assert_error(@() matsdca_loss(z,[2 3],struct('loss','multilabel_cross_entropy')));
assert_error(@() matsdca_loss(z,[2 3],struct('objective','cross_entropy')));
assert_error(@() matsdca_loss(z,[2 3],struct('alpha',1)));
assert_error(@() matsdca_loss(z,[2 3],struct('temperature',1)));
% precision: 'float32' or 'float64', by default the class of the scores.
for name={'cross_entropy','topk_cross_entropy','topk_hinge_alpha','multilabel_hinge','proximal_cross_entropy'}
  target=[2 3]; if strncmp(name{1},'multilabel',10), target=mask; end
  opts=struct('loss',name{1});
  if ~isempty(strfind(name{1},'topk')), opts.k=2; end
  if ~isempty(strfind(name{1},'hinge')), opts.gamma=.5; end
  [vs,gs]=matsdca_loss(single(z),target,opts);
  [vd,gd]=matsdca_loss(single(z),target,setfield(opts,'precision','float64'));
  [v32,g32]=matsdca_loss(single(z),target,setfield(opts,'precision','float32'));
  assert(isa(gs,'single') && isa(gd,'single') && max(abs(vs-vd)) < 1e-5 && max(abs(gs(:)-gd(:))) < 1e-5);
  assert(isequal(v32,vs) && isequal(g32,gs));
  assert_error(@() matsdca_loss(z,target,setfield(opts,'precision','float32')));
end
for alias={'single','double','fp32','fp64','float','half'}
  assert_error(@() matsdca_loss(z,[2 3],struct('precision',alias{1})));
  assert_error(@() matsdca_fit(X,y,struct('precision',alias{1})));
end
% opts.num_threads: any thread count gives the results of one thread.
Z=5*sin(reshape(1:64*2050,64,2050)); labels=mod(0:2049,64)+1;
M=Z>2; M(1,:)=true; M(2,:)=false;
for name={'cross_entropy','topk_hinge_beta','multilabel_hinge'}
  target=labels; if strncmp(name{1},'multilabel',10), target=M; end
  opts=struct('loss',name{1});
  if strcmp(name{1},'topk_hinge_beta'), opts.k=3; end
  if ~isempty(strfind(name{1},'hinge')), opts.gamma=.5; end
  [v1,g1]=matsdca_loss(Z,target,setfield(opts,'num_threads',1));
  [vs,gs]=matsdca_loss(single(Z),target,setfield(opts,'num_threads',1));
  [v0,g0]=matsdca_loss(Z,target,opts);
  assert(isequal(v0,v1) && isequal(g0,g1));
  for t=[2 7]
    [vd,gd]=matsdca_loss(Z,target,setfield(opts,'num_threads',t));
    [vt,gt]=matsdca_loss(single(Z),target,setfield(opts,'num_threads',t));
    assert(isequal(vd,v1) && isequal(gd,g1) && isequal(vt,vs) && isequal(gt,gs));
  end
end
% Invalid labels and nonfinite scores are rejected before any work.
bad=labels; bad(1900)=0; Zbad=Z; Zbad(3,700)=NaN;
assert_error(@() matsdca_loss(Z,bad,struct('num_threads',4)));
try
  v=matsdca_loss(Zbad,labels,struct('num_threads',4)); assert(false);
catch err
  assert(~isempty(strfind(err.message,'Scores must be finite')));
end
assert_error(@() matsdca_loss(z,1,struct('num_threads',0)));
assert_error(@() matsdca_loss(z,1,struct('num_threads',-1)));
% matsdca_fit: single data computes in float32 by default; 'float64' keeps double iterates.
o=struct('max_epoch',5);
m1=matsdca_fit(single(X),y,o);
m2=matsdca_fit(single(X),y,setfield(o,'precision','float32'));
m3=matsdca_fit(single(X),y,setfield(o,'precision','float64'));
assert(isequal(m1.A,m2.A) && strcmp(m1.precision,'float32') && strcmp(m3.precision,'float64') && isa(m3.A,'single'));
assert_error(@() matsdca_fit(X,y,setfield(o,'precision','float32')));
% eval_epoch = 0 evaluates once, at the end.
model=matsdca_fit(X,y,struct('eval_epoch',0,'max_epoch',5,'tol',0));
assert(model.epochs == 5 && numel(model.history) == 1 && strcmp(model.status,'max_epoch'));
% recompute_primal is read at run time; the duals do not depend on it.
on=matsdca_fit(X,y,struct('eval_epoch',5,'max_epoch',5));
off=matsdca_fit(X,y,struct('eval_epoch',5,'max_epoch',5,'recompute_primal',false));
assert(on.recompute_primal && ~off.recompute_primal && isequal(on.A, off.A));
assert(on.history(end).dual == off.history(end).dual);
% seed and num_threads: echoed; any seed solves.
for seed=[0 1 12345]
  ms=matsdca_fit(X,y,struct('seed',seed,'num_threads',2,'tol',1e-8));
  assert(ms.seed == seed && ms.num_threads == 2 && strcmp(ms.status,'solved'));
end
assert_error(@() matsdca_fit(X,y,struct('seed',-1)));
assert_error(@() matsdca_fit(X,y,struct('num_threads',0)));
% Projection work may change the solver trajectory; both fits solve to the same minimizer.
n=numel(y); M=full(sparse(y(:)',1:n,1,3,n)); M(sub2ind(size(M),mod(y(:)',3)+1,1:n))=1;
o=struct('loss','multilabel_hinge','gamma',.5,'tol',1e-10);
m64=matsdca_fit(X,sparse(M),o);
mext=matsdca_fit(X,sparse(M),setfield(o,'projection_precision','extended'));
assert(strcmp(m64.projection_precision,'float64') && strcmp(mext.projection_precision,'extended'));
assert(strcmp(m64.status,'solved') && strcmp(mext.status,'solved'));
e64=m64.history(end); eext=mext.history(end);
r64=sqrt(2*max(0,e64.relative_gap)*max(abs([e64.primal e64.dual])));
rext=sqrt(2*max(0,eext.relative_gap)*max(abs([eext.primal eext.dual])));
assert(norm(m64.W(:)-mext.W(:)) <= r64+rext+32*eps*max(1,norm(m64.W(:))));
zm=[.3 -1; -.8 .4; 1.2 .2; .1 .5]; mm=logical([1 0; 1 1; 0 0; 0 0]);
lo=struct('loss','multilabel_hinge','gamma',.5);
[v64,g64]=matsdca_loss(zm,mm,lo);
[vext,gext]=matsdca_loss(zm,mm,setfield(lo,'projection_precision','extended'));
assert(max(abs(v64-vext)) < 1e-14 && max(abs(g64(:)-gext(:))) < 1e-14);
assert_error(@() matsdca_fit(X,sparse(M),setfield(o,'projection_precision','double')));
assert_error(@() matsdca_loss(zm,mm,setfield(lo,'projection_precision','float32')));
% shrinking (hinge losses): 'auto' by default; every mode solves to the same objective.
o=struct('loss','multiclass_hinge','tol',1e-8,'max_epoch',20000);
ref=matsdca_fit(X,y,setfield(o,'shrinking','off'));
assert(strcmp(ref.shrinking,'off') && strcmp(ref.status,'solved'));
for s={'heuristic','safe','both','auto'}
  ms=matsdca_fit(X,y,setfield(o,'shrinking',s{1}));
  assert(strcmp(ms.shrinking,s{1}) && strcmp(ms.status,'solved'));
  assert(abs(ms.history(end).primal-ref.history(end).primal) <= 2e-8*abs(ref.history(end).primal));
end
assert(strcmp(matsdca_fit(X,y,struct('loss','multiclass_hinge','max_epoch',2)).shrinking,'auto'));
assert_error(@() matsdca_fit(X,y,setfield(o,'shrinking','on')));
% matsdca_prox at extreme scales: tiny biased knapsack, saturated bounds, 1e308 cone/simplex.
p_tiny=matsdca_prox([1e-20;1e-20],struct('prox','knapsack_le_biased','lo',0,'hi',1,'rhs',1,'rho',1));
assert(max(abs(p_tiny - 1e-20/3)) <= 16*eps*1e-20);
for prox_name={'knapsack','knapsack_le','knapsack_le_biased'}
  p_signed=matsdca_prox([-1e25;0;1e25],struct('prox',prox_name{1},'lo',-1e20,'hi',1e20,'rhs',-1,'rho',1));
  assert(isequal(p_signed,[-1e20;-1;1e20]));
  p_pos=matsdca_prox([1e25;-1e25;2],struct('prox',prox_name{1},'lo',-1e20,'hi',1e20,'rhs',1,'rho',1));
  assert(isequal(p_pos,[1e20;-1e20;1]));
end
assert(isequal(matsdca_prox([1e308;1e308],struct('prox','topk_cone','k',1)),[1e308;1e308]));
for prox_name={'topk_cone_biased','topk_simplex_biased','knapsack_le_biased'}
  p_big=matsdca_prox([1e308;1e308],struct('prox',prox_name{1},'k',1,'hi',1e308,'rhs',1e308,'rho',1));
  assert(max(abs(p_big - 1e308/3)) <= 16*eps*1e308);
end
assert_error(@() matsdca_prox([.7;-.2;.3],struct('prox','knapsack','alias',1)));
% matsdca_loss at extreme scales: tiny hinge margin, huge gamma, proximal_entropy cancellation.
[v_s4,g_s4]=matsdca_loss([1;1e-20],1,struct('loss','topk_hinge_beta','k',1,'gamma',1e-20));
assert(abs(v_s4/(0.5e-20) - 1) < 1e-13 && max(abs(g_s4 - [-1;1])) < 1e-13);
[v_s6,g_s6]=matsdca_loss([0;1e200],1,struct('loss','topk_hinge_beta','k',1,'gamma',1e300));
assert(abs(v_s6/5e99 - 1) < 1e-13 && abs(g_s6(2)/1e-100 - 1) < 1e-13);
[v_f4,~]=matsdca_loss([1e16;0;0],1,struct('loss','proximal_cross_entropy','alpha',1e50));
assert(v_f4 >= 0 && v_f4 < 1e-30);
end

function test_warm_start()
% Resuming a solved fit from its duals alone stays solved; W = center + X * A'.
X=[1 0 -1; 0 1 -1]; y=[1 2 3];
opts=struct('tol',1e-12,'eval_epoch',1);
for center={[], [.5 -1 .2; .3 .1 -.4]}
  if ~isempty(center{1}), opts.center=center{1}; end
  cold=matsdca_fit(X,y,opts);
  assert(strcmp(cold.status,'solved'));
  warm=opts; warm.initial_dual=cold.A; warm.max_epoch=1;
  model=matsdca_fit(X,y,warm);
  assert(model.epochs == 1 && abs(model.relative_gap) < 1e-10);
  W0=zeros(size(model.W)); if ~isempty(center{1}), W0=center{1}; end
  assert(max(max(abs(model.W - W0 - X*model.A'))) < 1e-12);
  assert(max(abs(model.W(:) - cold.W(:))) < 1e-5);  % the gap bounds the distance squared
end
assert_error(@() matsdca_fit(X,y,struct('initial_dual',zeros(3,2))));
assert_error(@() matsdca_fit(X,y,struct('initial_dual',[NaN 0 0; 0 0 0; 0 0 0])));
% Infeasible duals: a positive non-label entry; a column that does not sum to zero.
assert_error(@() matsdca_fit(X,y,struct('initial_dual',[-1 0 0; .5 0 0; .5 0 0])));
assert_error(@() matsdca_fit(X,y,struct('initial_dual',[-.5 0 0; .5 0 0; .1 0 0])));
end

function test_sparse_targets(X)
quiet=struct('loss','multilabel_hinge');
labels=[1 2 3 1]; n=numel(labels);
reference=matsdca_fit(X,sparse(labels,1:n,1,3,n),quiet);
% Spare capacity (nzmax > nnz), logical and full logical targets.
Y=spalloc(3,n,20);
for i=1:n, Y(labels(i),i)=1; end
assert(nzmax(Y) > nnz(Y));
spare=matsdca_fit(X,Y,quiet);
binary=matsdca_fit(X,logical(Y),quiet);
dense=matsdca_fit(X,full(logical(Y)),quiet);
assert(isequal(spare.W, reference.W) && isequal(binary.W, reference.W) && isequal(dense.W, reference.W));
% Classes without positives keep their rows: 2 and 3 of 4 here.
Y=sparse([2 3 3 2],1:n,1,4,n);
for model={matsdca_fit(X,Y,quiet), matsdca_fit({X,X},{Y,sparse(2,1:n,1,4,n)},quiet)}
  assert(model{1}.num_classes == 4 && isequal(size(model{1}.W),[2 4]));
  [~,predicted]=max(model{1}.W'*X);
  assert(all(predicted == [2 3 3 2]));
end
% Single labels: 1-based; num_classes keeps classes without examples.
assert_error(@() matsdca_fit(X,labels-1));
model=matsdca_fit(X,labels,struct('num_classes',5));
assert(model.num_classes == 5 && isequal(size(model.W),[2 5]));
end

function assert_error(f)
failed=false;
try
  result=f(); %#ok<NASGU>
catch
  failed=true;
end
assert(failed,'Expected invalid input to raise an error');
end

function test_prox_precision()
% Precision selects arithmetic independently of the stored input/output class.
for cast_input={@single,@double}
  x=cast_input{1}([.7;-.2;.3]);
  for name={'float32','float64','long_double'}
    actual=matsdca_prox(x,struct('prox','knapsack','precision',name{1}));
    assert(isa(actual,class(x)) && abs(sum(actual)-1) < 1e-6);
  end
  assert(isequal(matsdca_prox(x,struct('prox','knapsack')), ...
    matsdca_prox(x,struct('prox','knapsack','precision',strrep(strrep(class(x),'single','float32'),'double','float64')))));
end
for alias={'single','double','fp32','fp64','half'}
  assert_error(@() matsdca_prox([.7;-.2;.3],struct('precision',alias{1})));
end
end

function test_ordinary_inputs()
% Several examples (columns) of every hinge and of cross-entropy against closed forms.
m=4; n=5; t=0:m*n-1; Z=2*sin(0.37*reshape(t.^2+3*t,m,n)); y=mod(3*(0:n-1)+1,m)+1;
M=false(m,n); M(sub2ind([m n],y,1:n))=true; M(sub2ind([m n],mod(y,m)+1,1:n))=true;
cases={'cross_entropy',1; 'multiclass_hinge',1; 'topk_hinge_alpha',2; 'topk_hinge_beta',2; ...
  'multilabel_hinge',1};
for c=1:size(cases,1)
  name=cases{c,1}; k=cases{c,2}; target=y;
  if strcmp(name,'multilabel_hinge'), target=M; end
  opts=struct('loss',name);
  if ~isempty(strfind(name,'topk')), opts.k=k; end
  [v,g]=matsdca_loss(Z,target,opts);
  assert(isequal(size(v),[1 n]) && isequal(size(g),[m n]));
  for i=1:n
    z=Z(:,i); e=zeros(m,1);
    switch name
      case 'cross_entropy'
        p=exp(z-max(z)); p=p/sum(p); value=-log(p(y(i))); e=p; e(y(i))=e(y(i))-1;
      case 'multilabel_hinge'
        pos=find(M(:,i)); neg=find(~M(:,i));
        [lo,a]=min(z(pos)); [hi,b]=max(z(neg)); value=max(0,1+hi-lo);
        if value>0, e(neg(b))=1; e(pos(a))=-1; end
      otherwise
        others=setdiff(1:m,y(i)); [h,order]=sort(1+z(others)-z(y(i)),'descend');
        top=others(order(1:k)); h=h(1:k);
        if strcmp(name,'topk_hinge_beta')
          value=sum(max(h,0))/k; active=h>0;
        else
          value=max(0,mean(h)); active=repmat(value>0,k,1);
        end
        e(top(active))=1/k; e(y(i))=-sum(active)/k;
    end
    assert(abs(v(i)-value) < 1e-14 && max(abs(g(:,i)-e)) < 1e-14, name);
  end
end
% Every objective, nonsmooth and smoothed, solves at the primal it reports:
% W = X A' and 0.5 ||W||^2 + C sum(matsdca_loss(W' X)).
d=3; n=12; X=sin(0.61*reshape((0:d*n-1)*7+2,d,n)); y=mod(0:n-1,4)+1;
Y=sparse(y,1:n,1,4,n); Y(3,y==1)=1;
names={'cross_entropy','topk_cross_entropy','multiclass_hinge','topk_hinge_alpha', ...
  'topk_hinge_beta','multilabel_hinge','multilabel_cross_entropy'};
for i=1:numel(names)
  hinge=~isempty(strfind(names{i},'hinge'));
  for gamma=[0 .5]
    if gamma > 0 && ~hinge, continue; end
    multilabel=strncmp(names{i},'multilabel',10);
    target=y; scores=y;
    if multilabel, target=Y; scores=full(Y)>0; end
    params=struct('loss',names{i});
    if ~isempty(strfind(names{i},'topk')), params.k=2; end
    if hinge, params.gamma=gamma; end
    opts=params; opts.C=2; opts.tol=1e-8; opts.max_epoch=100000;
    model=matsdca_fit(X,target,opts);
    assert(strcmp(model.status,'solved'), names{i});
    assert(max(max(abs(model.W - X*model.A'))) < 1e-12);
    v=matsdca_loss(model.W'*X,scores,params);
    primal=0.5*sum(model.W(:).^2)+2*sum(v);
    assert(abs(primal-model.history(end).primal) <= 1e-12*abs(primal), names{i});
    assert(primal-model.history(end).dual <= 1e-8*abs(primal), names{i});
  end
end
end
