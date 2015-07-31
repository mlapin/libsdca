%
% Having fun with W_0(exp(x))
%

clear;
close all;

x1 = 1:1000;
w1 = zeros(size(x1));
for i = 1:numel(w1)
  w1(i) = fzero(@(w)w+log(w)-x1(i),x1(i));
end

x2 = -1000:-1;
w2 = zeros(size(x2));
for i = 1:numel(w2)
  w2(i) = fzero(@(w)w*exp(w)-exp(x2(i)),exp(x2(i)));
end

figure; grid on;
plot(x1, w1);

figure; grid on;
plot(x2, log10(w2));

figure; grid on;
plot(w1-x1);

figure; grid on;
plot(w2-x2);
