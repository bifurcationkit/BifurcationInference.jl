include("gradients.jl")
include("utils.jl")

############################################################## F(z,θ) = 0 region definition
F( z::AbstractVector, θ::AbstractVector ) = F( z[Not(end)], z[end], θ )
function F( u::AbstractVector, p::Number, θ::AbstractVector )

	f = first(u)*first(p)*first(θ)
	F = similar(u,typeof(f))
	a,b = θ

	F[1] = u[1] - 1/3*u[1]^3 - u[2] + p # exp((p+2)^2/3 +(u[1]+2)^2/3)
	F[2] = u[1] - a - b*u[2]/4

	return F
end

# integrand under dz
integrand( z::AbstractVector, θ::AbstractVector ) = integrand( z[Not(end)], z[end], θ )
function integrand( u::AbstractVector, p::Number, θ::AbstractVector; β=10 )
	z = [u;p]
	return (1-σ(β*(p-4)))*σ(β*(p+4))#exp(-norm(z)^2)#*(1-σ(β*(p-2)))*σ(β*(p+2))
end

using Flux: σ
u = [1.0,1.0] # initial root to peform continuation from
unit_test(xlim=(-1,1),ylim=(0,2))