######################################################## model
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
X = StateSpace( 2, 0.01:0.01:10, [4,5] )
θ = SizedVector{5}(0.0,0.0,-1.0,-1.0,2.0)