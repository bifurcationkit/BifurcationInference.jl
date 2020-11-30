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

	z = [u;p]
	R = [ cos(θ[2]) sin(θ[2]); -sin(θ[2]) cos(θ[2]) ]
	F[1] = exp(-norm(R*z.-1)^2) + exp(-norm(R*z.+1)^2) - θ[1]
	return F
end

# integrand under dz
integrand( z::AbstractVector, θ::AbstractVector ) = integrand( z[Not(end)], z[end], θ )
function integrand( u::AbstractVector, p::Number, θ::AbstractVector; β=10 )
	return 1.0 #(1-σ(β*(p-2)))*σ(β*(p+2))
end

u = [2.0] # initial root to peform continuation from
unit_test(xlim=(0.1,0.4),ylim=(-π/4,π/4),ϵ=1e-5)
