######################################################## model
function rates(u::AbstractVector,parameters::NamedTuple)

	@unpack θ,p = parameters
	μ₁,μ₂, a₁,a₂, k = θ

	f = first(u)*first(p)*first(θ)
	F = similar(u,typeof(f))

	F[1] = ( a₁^2 + (p*u[2])^2 ) / ( 1 + (p*u[2])^2 ) - μ₁^2*u[1]
	F[2] = ( a₂^2 + (k*u[1])^2 ) / ( 1 + (k*u[1])^2 ) - μ₂^2*u[2]

	return F
end

######################################################### targets and initial guess
targetData = StateSpace( 2, 0:0.001:8, [4,5] )
θ = SizedVector{5}(1.75,0.1,1.0,1.0,3.0)
