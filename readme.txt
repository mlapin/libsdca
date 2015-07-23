mkdir build
cd build
cmake ..
make
make install


How to debug mex code with Qt Creator:
--------------------------------------
Debug / Start and Debug External Application... (Ctrl + F5)
Local executable: /usr/lib/matlab-8.4/bin/glnxa64/MATLAB
Command line arguments: -nojvm -r 'mexdebug; quit;'
Working directory: .../libsdca/matlab
Run in terminal: checked
Debug information: .../libsdca

How to profile mex code with Intel VTune Amplifier:
---------------------------------------------------
Application: /usr/lib/matlab-8.4/bin/glnxa64/MATLAB
Application parameters: -nojvm -r 'mexdebug; quit;'
Use application directory as working directory: not checked
Working directory: .../libsdca/matlab
