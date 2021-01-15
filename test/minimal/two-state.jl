######################################################## model
function rates(u::AbstractVector,parameters::NamedTuple)

	@unpack θ,p = parameters
	μ₁,μ₂ = θ

	f = first(u)*first(p)*first(θ)
	F = similar(u,typeof(f))

	F[1] = 1 / ( 1 + (p*u[2])^2 ) - μ₁*u[1]
	F[2] = 1 / ( 1 + (3*u[1])^2 ) - μ₂*u[2]

	return F
end

######################################################### targets and initial guess
targetData = StateSpace( 2, 0:0.005:8, [4,5] )
θ = SizedVector{2}(1.75,0.1)
