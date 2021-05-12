######################################################## model
function rates(u::AbstractVector,parameters::NamedTuple)

	@unpack θ,p = parameters
	μ₁,μ₂, a₁,a₂, k₁,k₂, ϵ₁ = θ

	f = first(u)*first(p)*first(θ)
	F = similar(u,typeof(f))

	F[1] = ( ϵ₁ + a₁*u[2]^2 ) / ( k₁^2 + u[2]^2 ) - μ₁*u[1]
	F[2] = ( p  + a₂*u[1]^2 ) / ( k₂^2 + u[1]^2 ) - μ₂*u[2]

	return F
end

######################################################### targets and initial guess
targetData = StateSpace( 2, 0:0.01:8, [4,5] )
θ = SizedVector{7}(4.0,4.0, 0.0,0.0, 1.0,1.0, 5.0)