%
% Plot Lambert W of exp
%

clc;
clear;
close all;
% addpath(fullfile(pwd, 'libsdca-debug'));
addpath(fullfile(pwd, 'libsdca-release'));
addpath('/BS/mlapin-projects3/work/cvpr16/code/src/utility');
rng(0);


opts.prox = 'lambert_w_exp';

t1 = 0:0.1:10;
x1 = libsdca_prox(t1, opts);

t2 = -10:0.1:0;
x2 = libsdca_prox(t2, opts);

colors = my5colors;

figure; grid on; hold on;
plot(t1, x1, '.-', 'LineWidth', 2, 'Color', colors(1,:));
plot(t1, t1 - log(t1), '.-', 'LineWidth', 2, 'Color', colors(3,:));
plot(t1, t1, '.-', 'LineWidth', 2, 'Color', colors(2,:));
legend('Lambert W(exp(t))', 't - log(t)', 't', 'Location', 'NorthWest');
xlabel('t');
ylabel('Linear scale');
set(gca, 'Xtick', 0:10);
set(gca, 'Ytick', 0:10);
axis([0 10 0 10]);

set(gca, 'FontSize', 18);
printpdf('lambert-w-exp-pos');

close all;

figure; grid on; hold on;
plot(t2, log(x2), '.-', 'LineWidth', 2, 'Color', colors(1,:));
plot(t2, t2, '.-', 'LineWidth', 2, 'Color', colors(3,:));
legend('Lambert W(exp(t))', 'exp(t)', 'Location', 'NorthWest');
xlabel('t');
ylabel('Log scale');
set(gca, 'Xtick', -10:0);
set(gca, 'Ytick', -9:0);
axis([-10 0 -10 0]);

set(gca, 'FontSize', 18);
printpdf('lambert-w-exp-neg');
