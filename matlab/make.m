%
% Make script for matsdca
%

% Library version and the names for the mex files
LIBSDCA_VERSION = '0.3.1';
MEX_PROX = 'matsdca_prox';
MEX_SOLVE = 'matsdca_fit';

% Uncomment the -v for debugging
CXXFLAGS = {'\$CXXFLAGS', '-std=c++11', ... % '-v', ...
  '-I../include', '-I../src/matlab', '-DBLAS_MATLAB', ...
  sprintf('-DLIBSDCA_VERSION=''"%s"''', LIBSDCA_VERSION), ...
  sprintf('-DMEX_PROX=''"%s"''', MEX_PROX), ...
  sprintf('-DMEX_SOLVE=''"%s"''', MEX_SOLVE), ...
  };
CXXFLAGS = sprintf('CXXFLAGS="%s"', sprintf('%s ', CXXFLAGS{:}));

old_pwd = pwd;
try
  cd(fileparts(mfilename('fullpath')));
	mex(CXXFLAGS,'-largeArrayDims', '../src/matlab/mex_prox.cpp', ...
    '-output', MEX_PROX);
  mex(CXXFLAGS,'-largeArrayDims', '../src/matlab/mex_solve.cpp', ...
    '-output', MEX_SOLVE, '-lmwblas');
catch me
  disp(getReport(me));
	fprintf('Check the configured compiler with `mex -setup`.\n');
  fprintf('If the problem persists, try the CMake build.\n');
end
cd(old_pwd);
