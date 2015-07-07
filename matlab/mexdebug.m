clear;
close all;
addpath('libsdca');

d = 10;
n = 5;

opts.proj = 'topk_simplex';
opts.rhs = .75;
opts.rho = 1.5;
opts.k = 3;

A = randn(d,n);
B = libsdca_prox(A, opts);
