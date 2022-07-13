# Fractional exponential sums

This repository contains the code for the numerical experiments in the paper "Low-rank tensor structure preservation in fractional
operators by means of exponential sums"; the experiments are briefly described below. Please note that some of them require the 
use of the [Tensor Toolbox](https://www.tensortoolbox.org/), [TensorLab](https://www.tensorlab.net/), or the [TT-Toolbox](https://github.com/oseledets/TT-Toolbox). 

## Description of the experiments

### Solution of a tensor equation
The example ```num_exp_dense.m``` solves a fractional tensor equation in 3 dimensions by exponential sums, 
and compares the result with the dense solution obtained by diagonalization. 

### Solution of a low CP-rank equation
The example ```num_exp_lowrank.m``` deals with the case above, but assumes that the right hand side in the equation is 
product of three functions that depend on x, y, and z, respectively. Then, the right hand side is stored in the CP format, 
and the solution is computed in the same format by exponential sum approximation. 

### Low-rank approximability properties
The test ```num_exp_lowrank_approximability.m``` checks the bounds for the low-rank approximability in CP, TT, and 
Tucker formats for the solution of fractional differential equations. 

### High-dimensional TT solver 
In the last test, ```num_exp_tt.m```, a high-dimensional equation over [0, 1]^d (up to $d = 20$) is solved by combining 
a TT-cross approximation for the right hand side, and the exponential sum approximation in the TT-format. 
