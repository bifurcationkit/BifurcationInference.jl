# BifurcationInference.jl

This library implements the method described in **Szep, G. Dalchau, N. and Csikasz-Nagy, A. 2021. [Parameter Inference with Bifurcation Diagrams](https://arxiv.org/abs/2106.04243)** using parameter continuation library [`BifurcationKit.jl`](https://github.com/rveltz/BifurcationKit.jl) and auto-differentiation [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl). This implementation enables continuation methods can be used as layers in machine learning proceedures, and inference can be run end-to-end directly on the geometry of state space.

[![Build Status](https://travis-ci.com/gszep/BifurcationInference.jl.svg?branch=master)](https://travis-ci.com/gszep/BifurcationInference.jl)
[![Coverage](https://codecov.io/gh/gszep/BifurcationInference.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/gszep/BifurcationInference.jl)
[![arXiv](https://img.shields.io/badge/arXiv-2106.04243-b31b1b.svg)](https://arxiv.org/abs/2106.04243)

## Basic Usage
The model definition requires a distpatched method on `F(z::BorderedArray,θ::AbstractVector)` where `BorderedArray` is a type that contains the state vector `u` and control condition `p` used by the library [`BifurcationKit.jl`](https://github.com/rveltz/BifurcationKit.jl). `θ` is a vector of parameters to be optimised.
```julia
using BifurcationInference, StaticArrays

F(z::BorderedArray,θ::AbstractVector) = F(z.u,(θ=θ,p=z.p))
function F(u::AbstractVector,parameters::NamedTuple)

	@unpack θ,p = parameters
	μ₁,μ₂, a₁,a₂, k = θ

	f = first(u)*first(p)*first(θ)
	F = similar(u,typeof(f))

	F[1] = ( 10^a₁ + (p*u[2])^2 ) / ( 1 + (p*u[2])^2 ) - u[1]*10^μ₁
	F[2] = ( 10^a₂ + (k*u[1])^2 ) / ( 1 + (k*u[1])^2 ) - u[2]*10^μ₂

	return F
end
```
The targets are specified with `StateSpace( dimension::Integer, condition::AbstractRange, targets::AbstractVector )`. It contains the dimension of the state space, which must match that of the defined model, the control condition range that we would like to perform the continuation method in, and a vector of target conditions we would like to match.
```julia
X = StateSpace( 2, 0:0.01:10, [4,5] )
```
The optimisation needs to be initialised using a `NamedTuple` containing the initial guess for `θ` and the initial value `p` from which to begin the continuation.
```julia
using Flux: Optimise
parameters = ( θ=SizedVector{5}(0.5,0.5,0.5470,2.0,7.5), p=minimum(X.parameter) )
train!( F, parameters, X;  iter=200, optimiser=Optimise.ADAM(0.01) )
```
