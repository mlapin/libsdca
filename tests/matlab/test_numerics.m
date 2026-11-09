function test_numerics
% Run after make.m. Scores are classes-by-examples; labels are 1-based.
z = [20; -20; -20];
[v,g] = matsdca_loss(z,1);
assert(v > 0 && g(1) < 0);
assert(abs(sum(g)) < 1e-30);
v = matsdca_loss(ones(3,1)*1e16,1);
assert(abs(v-log(3)) < 1e-14);
names = {'cross_entropy','topk_cross_entropy','topk_hinge_alpha','topk_hinge_beta', ...
  'topk_hinge_alpha','topk_hinge_beta','multilabel_hinge', ...
  'multilabel_hinge','multilabel_cross_entropy','proximal_cross_entropy'};
z = [.3; -.8; 1.2; .1];
for i=1:numel(names)
  opts = struct('objective',names{i},'k',2,'gamma',.7,'alpha',1.2);
  target = 1;
  if startsWith(names{i},'multilabel'), target=logical([1;1;0;0]); end
  [~,g] = matsdca_loss(z,target,opts);
  for j=1:numel(z)
    plus=z; minus=z; plus(j)=plus(j)+1e-5; minus(j)=minus(j)-1e-5;
    numeric=(matsdca_loss(plus,target,opts)-matsdca_loss(minus,target,opts))/2e-5;
    assert(abs(numeric-g(j)) < 1e-6);
  end
end
% A tiny smoothing parameter still enforces the total projection mass.
for gamma=[1e-10,1e-20,1e-100]
  for k=1:3
    [v,g]=matsdca_loss(zeros(4,1),1, ...
      struct('objective','topk_hinge_beta','k',k,'gamma',gamma));
    assert(abs(v-(1-gamma/6)) < 2e-15);
    assert(max(abs(g-[-1;1/3;1/3;1/3])) < 2e-15);
  end
end
% Shared option and objective names match the historical gateways.
m1=matsdca_fit(eye(3),[1;2;3],struct('objective','softmax','c',2,'epsilon',1e-6));
m2=matsdca_fit(eye(3),[1;2;3],struct('objective','cross_entropy','C',2,'tol',1e-6));
assert(max(abs(m1.W(:)-m2.W(:))) < 1e-14);
q=matsdca_prox(-20,struct('prox','topk_entropy_biased','alpha',1));
assert(q > 0);
assert_error(@() matsdca_loss(z,1,struct('k',1.5)));
assert_error(@() matsdca_loss(z,0));
assert_error(@() matsdca_prox(z,struct('k',-1)));
% Exercise every multiclass/multilabel solver dispatch on a small problem.
X=[1 0 -1 .5; 0 1 .5 -1]; y=[1 2 3 1];
objectives={'softmax','msvm','msvm_smooth','topk_hinge_alpha', ...
  'topk_hinge_beta','l2_entropy_topk','mlsoftmax','mlsvm','mlsvm_smooth'};
for i=1:numel(objectives)
  target=y;
  if startsWith(objectives{i},'ml')
    target=sparse(y,1:numel(y),ones(size(y)),3,numel(y));
  end
  model=matsdca_fit(X,target,struct('objective',objectives{i},'max_epoch',1000));
  assert(isstruct(model));
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
% Every loss checks 1 <= k < classes (3 classes here), also where it does not use k.
quiet=struct('log_level','none');
Y=sparse(y,1:numel(y),1,3,numel(y));
z=[.3 -1 .2 .1; -.8 .4 .5 -.1; 1.2 .2 -.3 .4];
names={'cross_entropy','topk_cross_entropy','multiclass_hinge','topk_hinge_alpha', ...
  'topk_hinge_beta','multilabel_hinge','multilabel_cross_entropy','proximal_cross_entropy'};
for i=1:numel(names)
  multilabel=strncmp(names{i},'multilabel',10);
  target=y; scores=y;
  if multilabel, target=Y; scores=full(Y)>0; end
  fit=~strcmp(names{i},'proximal_cross_entropy');
  for k=[0 3]
    assert_error(@() matsdca_loss(z,scores,struct('loss',names{i},'k',k)));
    if fit
      assert_error(@() matsdca_fit(X,target,setfield(setfield(quiet,'objective',names{i}),'k',k)));
    end
  end
  assert(all(isfinite(matsdca_loss(z,scores,struct('loss',names{i},'k',2)))));
  if fit
    assert(isstruct(matsdca_fit(X,target,setfield(setfield(quiet,'objective',names{i}),'k',2))));
  end
end
end

function test_names(X,y)
% The model reports the objective name as given; losses take opts.loss.
quiet=struct('log_level','none');
model=matsdca_fit(X,y,quiet);
assert(strcmp(model.objective,'cross_entropy'));
for name={'cross_entropy','softmax','topk_hinge_alpha','topk_hinge_beta'}
  model=matsdca_fit(X,y,setfield(quiet,'objective',name{1}));
  assert(strcmp(model.objective,name{1}));
end
Y=sparse(y,1:numel(y),1,3,numel(y));
model=matsdca_fit(X,Y,setfield(quiet,'objective','multilabel_cross_entropy'));
assert(strcmp(model.objective,'multilabel_cross_entropy'));
z=[.3 -1; -.8 .4; 1.2 .2; .1 .5]; mask=logical([1 0; 1 1; 0 0; 0 0]);
[v1,g1]=matsdca_loss(z,mask,struct('loss','multilabel_cross_entropy'));
[v2,g2]=matsdca_loss(z,mask,struct('objective','cross_entropy'));
assert(isequal(v1,v2) && isequal(g1,g2));
assert(isequal(matsdca_loss(z,[2 3],struct('loss','topk_hinge_beta','k',2)), ...
  matsdca_loss(z,[2 3],struct('objective','topk_hinge_beta','k',2))));
assert_error(@() matsdca_loss(z,[2 3],struct('loss','cross_entropy','objective','cross_entropy')));
% opts.precision = 'single' computes single scores in float.
for name={'cross_entropy','topk_cross_entropy','topk_hinge_alpha','multilabel_hinge','proximal_cross_entropy'}
  target=[2 3]; if strncmp(name{1},'multilabel',10), target=mask; end
  opts=struct('loss',name{1},'k',2,'gamma',.5);
  [vs,gs]=matsdca_loss(single(z),target,setfield(opts,'precision','single'));
  [vd,gd]=matsdca_loss(single(z),target,opts);
  assert(isa(gs,'single') && max(abs(vs-vd)) < 1e-5 && max(abs(gs(:)-gd(:))) < 1e-5);
  % The names shared with the other interfaces select the same arithmetic.
  [v32,g32]=matsdca_loss(single(z),target,setfield(opts,'precision','float32'));
  [v64,g64]=matsdca_loss(single(z),target,setfield(opts,'precision','float64'));
  assert(isequal(v32,vs) && isequal(g32,gs) && isequal(v64,vd) && isequal(g64,gd));
end
assert_error(@() matsdca_loss(z,[2 3],struct('precision','half')));
% opts.num_threads: any thread count gives the results of one thread.
Z=5*sin(reshape(1:64*2050,64,2050)); labels=mod(0:2049,64)+1;
M=Z>2; M(1,:)=true; M(2,:)=false;
for name={'cross_entropy','topk_hinge_beta','multilabel_hinge'}
  target=labels; if strncmp(name{1},'multilabel',10), target=M; end
  opts=struct('loss',name{1},'k',3,'gamma',.5);
  [v1,g1]=matsdca_loss(Z,target,setfield(opts,'num_threads',1));
  [vs,gs]=matsdca_loss(single(Z),target,setfield(opts,'num_threads',1));
  [v0,g0]=matsdca_loss(Z,target,opts);
  assert(isequal(v0,v1) && isequal(g0,g1));
  for t=[0 2 7]
    [vd,gd]=matsdca_loss(Z,target,setfield(opts,'num_threads',t));
    [vt,gt]=matsdca_loss(single(Z),target,setfield(opts,'num_threads',t));
    assert(isequal(vd,v1) && isequal(gd,g1) && isequal(vt,vs) && isequal(gt,gs));
  end
end
% The first failing example's error, as in a serial loop.
bad=labels; bad(1900)=0; Zbad=Z; Zbad(3,700)=NaN;
for t=[1 4]
  try
    v=matsdca_loss(Zbad,bad,struct('num_threads',t)); assert(false);
  catch err
    assert(~isempty(strfind(err.message,'Scores must be finite')));
  end
end
assert_error(@() matsdca_loss(z,1,struct('num_threads',-1)));
% matsdca_fit takes the same names: 'float32' is 'single', 'float64' is 'double'.
o=struct('log_level','none','max_epoch',5);
m1=matsdca_fit(single(X),y,setfield(o,'precision','single'));
m2=matsdca_fit(single(X),y,setfield(o,'precision','float32'));
m3=matsdca_fit(single(X),y,o);
m4=matsdca_fit(single(X),y,setfield(o,'precision','float64'));
assert(isequal(m1.A,m2.A) && isequal(m3.A,m4.A) && strcmp(m2.precision,'float32') && strcmp(m4.precision,'float64') && strcmp(m2.data_precision,'float32'));
assert_error(@() matsdca_fit(X,y,struct('precision','half')));
% 'float' is ambiguous across the interfaces (Python's float is float64): rejected.
assert_error(@() matsdca_fit(X,y,struct('precision','float')));
assert_error(@() matsdca_loss(z,1,struct('precision','float')));
assert_error(@() matsdca_prox([.7;-.2;.3],struct('prox','knapsack','precision','float')));
% eval_epoch = 0 evaluates once, at the end.
model=matsdca_fit(X,y,struct('log_level','none','eval_epoch',0,'max_epoch',5));
assert(model.epoch == 5 && numel(model.train) == 1);
% recompute_primal is read at run time; the duals do not depend on it.
on=matsdca_fit(X,y,struct('log_level','none','eval_epoch',5,'max_epoch',5));
off=matsdca_fit(X,y,struct('log_level','none','eval_epoch',5,'max_epoch',5,'recompute_primal',false));
assert(on.recompute_primal && ~off.recompute_primal && isequal(on.A, off.A));
assert(on.train(end).dual == off.train(end).dual);
% projection_precision: 'float64' by default, 'extended' agrees to rounding.
n=numel(y); M=full(sparse(y(:)',1:n,1,3,n)); M(sub2ind(size(M),mod(y(:)',3)+1,1:n))=1;
o=struct('log_level','none','objective','multilabel_hinge','gamma',.5,'epsilon',1e-10);
m64=matsdca_fit(X,sparse(M),o);
mext=matsdca_fit(X,sparse(M),setfield(o,'projection_precision','extended'));
assert(strcmp(m64.projection_precision,'float64') && strcmp(mext.projection_precision,'extended'));
assert(max(abs(m64.W(:)-mext.W(:))) < 1e-12);
zm=[.3 -1; -.8 .4; 1.2 .2; .1 .5]; mm=logical([1 0; 1 1; 0 0; 0 0]);
lo=struct('loss','multilabel_hinge','gamma',.5);
[v64,g64]=matsdca_loss(zm,mm,lo);
[vext,gext]=matsdca_loss(zm,mm,setfield(lo,'projection_precision','extended'));
assert(max(abs(v64-vext)) < 1e-14 && max(abs(g64(:)-gext(:))) < 1e-14);
assert_error(@() matsdca_fit(X,y,struct('projection_precision','double')));
assert_error(@() matsdca_loss(zm,mm,setfield(lo,'projection_precision','float32')));
% shrinking: 'auto' by default; every mode solves to the same objective.
o=struct('log_level','none','objective','multiclass_hinge','epsilon',1e-8,'max_epoch',20000);
ref=matsdca_fit(X,y,setfield(o,'shrinking','off'));
assert(strcmp(ref.shrinking,'off') && strcmp(ref.status,'solved'));
for s={'heuristic','safe','both','auto'}
  ms=matsdca_fit(X,y,setfield(o,'shrinking',s{1}));
  assert(strcmp(ms.shrinking,s{1}) && strcmp(ms.status,'solved'));
  assert(abs(ms.train(end).primal-ref.train(end).primal) <= 2e-8*abs(ref.train(end).primal));
end
assert(strcmp(matsdca_fit(X,y,struct('log_level','none','max_epoch',2)).shrinking,'auto'));
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
% matsdca_loss at extreme scales: tiny hinge margin, huge gamma, proximal_entropy cancellation.
[v_s4,g_s4]=matsdca_loss([1;1e-20],1,struct('loss','topk_hinge_beta','gamma',1e-20));
assert(abs(v_s4/(0.5e-20) - 1) < 1e-13 && max(abs(g_s4 - [-1;1])) < 1e-13);
[v_s6,g_s6]=matsdca_loss([0;1e200],1,struct('loss','topk_hinge_beta','gamma',1e300));
assert(abs(v_s6/5e99 - 1) < 1e-13 && abs(g_s6(2)/1e-100 - 1) < 1e-13);
[v_f4,~]=matsdca_loss([1e16;0;0],logical([1;1;0]),struct('loss','proximal_cross_entropy','alpha',1e50));
assert(abs(v_f4/log(2) - 1) < 1e-13);
end

function test_warm_start()
% Resuming a converged fit from its duals alone stays converged.
X=[1 0 -1; 0 1 -1]; y=[1 2 3];
opts=struct('log_level','none','tol',1e-12,'eval_epoch',1);
for center={[], [.5 -1 .2; .3 .1 -.4]}
  if ~isempty(center{1}), opts.center=center{1}; end
  cold=matsdca_fit(X,y,opts);
  assert(strcmp(cold.status,'solved'));
  warm=opts; warm.initial_dual=cold.A; warm.max_epoch=1; warm.eval_on_start=false;
  model=matsdca_fit(X,y,warm);
  assert(model.epoch == 1 && abs(model.relative_gap) < 1e-10);
  W0=zeros(size(model.W)); if ~isempty(center{1}), W0=center{1}; end
  assert(max(max(abs(model.W - W0 - X*model.A'))) < 1e-12);
  assert(max(abs(model.W(:) - cold.W(:))) < 1e-5);  % the gap bounds the distance squared
  % A given W is rebuilt from the duals.
  warm.W=zeros(size(cold.W));
  rebuilt=matsdca_fit(X,y,warm);
  assert(isequal(rebuilt.W, model.W));
end
assert_error(@() matsdca_fit(X,y,struct('initial_dual',zeros(3,2))));
assert_error(@() matsdca_fit(X,y,struct('initial_dual',[NaN 0 0; 0 0 0; 0 0 0])));
end

function test_sparse_targets(X)
quiet=struct('log_level','none','objective','multilabel_hinge');
labels=[1 2 3 1]; n=numel(labels);
reference=matsdca_fit(X,sparse(labels,1:n,1,3,n),quiet);
% Spare capacity (nzmax > nnz) and logical targets.
Y=spalloc(3,n,20);
for i=1:n, Y(labels(i),i)=1; end
assert(nzmax(Y) > nnz(Y));
spare=matsdca_fit(X,Y,quiet);
binary=matsdca_fit(X,logical(Y),quiet);
assert(isequal(spare.W, reference.W) && isequal(binary.W, reference.W));
% Classes without positives keep their rows: 2 and 3 of 4 here.
Y=sparse([2 3 3 2],1:n,1,4,n);
for model={matsdca_fit(X,Y,quiet), matsdca_fit({X,X},{Y,sparse(2,1:n,1,4,n)},quiet)}
  assert(model{1}.num_classes == 4 && isequal(size(model{1}.W),[2 4]));
  [~,predicted]=max(model{1}.W'*X);
  assert(all(predicted == [2 3 3 2]));
end
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
  for aliases={{'single','float32','fp32'},{'double','float64','fp64'}}
    names=aliases{1};
    expected=matsdca_prox(x,struct('prox','knapsack','precision',names{1}));
    for name=names
      actual=matsdca_prox(x,struct('prox','knapsack','precision',name{1}));
      assert(isa(actual,class(x)) && isequal(actual,expected));
    end
  end
end
assert_error(@() matsdca_prox([.7;-.2;.3],struct('precision','half')));
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
  [v,g]=matsdca_loss(Z,target,struct('loss',name,'k',k));
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
% Every objective, nonsmooth and smoothed, solves at the primal it reports: W = X A' and
% 0.5 ||W||^2 + C sum(matsdca_loss(W' X)); zero-based labels train the same model.
d=3; n=12; X=sin(0.61*reshape((0:d*n-1)*7+2,d,n)); y=mod(0:n-1,4)+1;
Y=sparse(y,1:n,1,4,n); Y(3,y==1)=1;
names={'cross_entropy','topk_cross_entropy','multiclass_hinge','topk_hinge_alpha', ...
  'topk_hinge_beta','multilabel_hinge','multilabel_cross_entropy'};
for i=1:numel(names)
  for gamma=[0 .5]
    multilabel=strncmp(names{i},'multilabel',10);
    target=y; scores=y;
    if multilabel, target=Y; scores=full(Y)>0; end
    opts=struct('log_level','none','objective',names{i},'k',2,'gamma',gamma,'C',2, ...
      'tol',1e-8,'max_epoch',100000);
    model=matsdca_fit(X,target,opts);
    assert(strcmp(model.status,'solved'), names{i});
    assert(max(max(abs(model.W - X*model.A'))) < 1e-12);
    v=matsdca_loss(model.W'*X,scores,struct('loss',names{i},'k',2,'gamma',gamma));
    primal=0.5*sum(model.W(:).^2)+2*sum(v);
    assert(abs(primal-model.train(end).primal) <= 1e-12*abs(primal), names{i});
    assert(primal-model.train(end).dual <= 1e-8*abs(primal), names{i});
    if ~multilabel
      zero=matsdca_fit(X,y-1,opts);
      assert(isequal(zero.A,model.A) && isequal(zero.W,model.W), names{i});
    end
  end
end
end
