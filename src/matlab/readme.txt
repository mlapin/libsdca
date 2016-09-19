Matlab interfaces:
------------------
mex_solve.cpp   - SDCA solver (the main entry point);
mex_prox.cpp    - proximal operators (for use outside of the solver).


Legacy code (not part of libsdca):
----------------------------------
mex_gd.cpp - gradient descent for the nonconvex top-k entropy (top-k softmax) loss.



How to debug mex code in Qt Creator:
------------------------------------
Debug > Start Debugging > Start and Debug External Application... (Ctrl + F5)

Local executable: /usr/lib/matlab-8.6/bin/glnxa64/MATLAB

Command line arguments: -nojvm -r 'mexdebug; quit;'

Working directory: matlab

Run in terminal: checked

Debug information: .


How to profile mex code with Intel VTune Amplifier:
---------------------------------------------------
Application: /usr/lib/matlab-8.6/bin/glnxa64/MATLAB

Application parameters: -nojvm -r 'mexdebug; quit;'

Use application directory as working directory: not checked

Working directory: matlab
