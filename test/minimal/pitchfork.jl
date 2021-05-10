######################################################## model
function rates(u::AbstractVector,parameters::NamedTuple)
	@unpack θ,p = parameters

	f = first(u)*first(p)*first(θ)
	F = similar(u,typeof(f))

	F[1] = θ[1] + p*u[1] + θ[2]*u[1]^3
	F[2] = u[1] - u[2] # dummy second dimension

	return F
end

######################################################### targets and initial guess
targetData = StateSpace( 2, -5:0.01:5, [0] )
hyperparameters = getParameters(targetData)
θ = SizedVector{2}(4.0,-π/2)
