using InvertedIndices,LinearAlgebra,Plots
using ForwardDiff,BifurcationKit
global Iterable = BifurcationKit.ContIterable

import LinearAlgebra: norm
using Setfield: @lens
using Flux: σ

include("gradients.jl")
include("utils.jl")

############################################################## F(z,θ) = 0 region definition
F( z::AbstractVector, θ::AbstractVector ) = F( z[Not(end)], z[end], θ )
function F( u::AbstractVector, p::Number, θ::AbstractVector )
	f = first(u)*first(p)*first(θ)
	F = similar(u,typeof(f))

	F[1] = u[1] - θ[1]*exp(-p^2) - θ[2]
	return F
end

# integrand under dz
integrand( z::AbstractVector, θ::AbstractVector ) = integrand( z[Not(end)], z[end], θ )
function integrand( u::AbstractVector, p::Number, θ::AbstractVector; β=10 )
	return (1-σ(β*(p-2)))*σ(β*(p+2))
end

u = [1.0] # initial root to peform continuation from
unit_test()
