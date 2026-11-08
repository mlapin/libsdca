%
% Make script for matsdca
%

% Library version and the names for the mex files
LIBSDCA_VERSION = '0.3.0';
MEX_PROX = 'matsdca_prox';
MEX_SOLVE = 'matsdca_fit';
MEX_LOSS = 'matsdca_loss';

% Uncomment the -v for debugging. The numerics flags match the CMake build.
CXXFLAGS = {'\$CXXFLAGS', '-std=c++11', '-fno-fast-math', '-ffp-contract=off', ... % '-v', ...
  '-I../include', '-I../src/matlab', '-DBLAS_MATLAB', ...
  sprintf('-DSDCA_VERSION=''"%s"''', LIBSDCA_VERSION), ...
  sprintf('-DMEX_PROX=''"%s"''', MEX_PROX), ...
  sprintf('-DMEX_SOLVE=''"%s"''', MEX_SOLVE), ...
  sprintf('-DMEX_LOSS=''"%s"''', MEX_LOSS), ...
  };
CXXFLAGS = sprintf('CXXFLAGS="%s"', sprintf('%s ', CXXFLAGS{:}));

old_pwd = pwd;
try
  cd(fileparts(mfilename('fullpath')));
	mex(CXXFLAGS,'-largeArrayDims', '../src/matlab/mex_prox.cpp', ...
    '-output', MEX_PROX);
  mex(CXXFLAGS,'-largeArrayDims', '../src/matlab/mex_solve.cpp', ...
    '-output', MEX_SOLVE, '-lmwblas');
  mex(CXXFLAGS,'-largeArrayDims', '../src/matlab/mex_loss.cpp', ...
    '-output', MEX_LOSS, '-lmwblas');
catch me
  disp(getReport(me));
	fprintf('Check the configured compiler with `mex -setup`.\n');
  fprintf('If the problem persists, try the CMake build.\n');
  cd(old_pwd);
  rethrow(me);
end
cd(old_pwd);
