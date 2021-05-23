# BifurcationFit.jl

This library implements the method described in **Szep, G. Dalchau, N. and Csikasz-Nagy, A. 2021. [Parameter Inference with Bifurcation Diagrams](https://github.com/gszep/BifurcationFit.jl/blob/master/docs/article.pdf)** using parameter continuation library [`BifurcationKit.jl`](https://github.com/rveltz/BifurcationKit.jl) and auto-differentiation [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl). This implementation enables continuation methods can be used as layers in machine learning proceedures, and inference can be run end-to-end directly on the geometry of state space.

[![Build Status](https://travis-ci.com/gszep/FluxContinuation.jl.svg?branch=master)](https://travis-ci.com/gszep/FluxContinuation.jl)
[![Coverage](https://codecov.io/gh/gszep/FluxContinuation.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/gszep/FluxContinuation.jl)

## Basic Usage

```julia
using BifurcationFit

######################################################## define the model
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

######################################################### targets and initial guess
X = StateSpace( 2, 0:0.01:10, [4,5] )
θ = SizedVector{5}(0.5,0.5,0.5470,2.0,7.5)

######################################################### optimise parameters
train!( F, (θ=θ,p=minimum(X.parameter)), X;  iter=200, optimiser=ADAM(0.01) )
```
