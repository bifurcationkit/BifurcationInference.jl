######################################################## model
function rates(u::AbstractVector,parameters::NamedTuple)

	@unpack θ,p = parameters
	μ₁,μ₂, a₁,a₂, ϵ₁,ϵ₂, k = θ

	f = first(u)*first(p)*first(θ)
	F = similar(u,typeof(f))

	F[1] = ( ϵ₁ + a₁*u[2]^2 ) / ( p^2 + u[2]^2 ) - μ₁*u[1]
	F[2] = ( ϵ₂ + a₂*u[1]^2 ) / ( k^2 + u[1]^2 ) - μ₂*u[2]

	return F
end

######################################################### targets and initial guess
targetData = StateSpace( 2, 0:0.001:8, [4,5] )
θ = SizedVector{7}(1.75,0.1,1.0,1.0,1.0,1.0,3.0)