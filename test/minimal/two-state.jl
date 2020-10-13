######################################################## model
function rates(u::AbstractVector,parameters::NamedTuple)

	@unpack θ,p = parameters
	μ₁,μ₂,k = θ

	f = first(u)*first(p)*first(θ)
	F = similar(u,typeof(f))

	F[1] = 1 / ( 1 + (p*u[2])^2 ) - μ₁*u[1]
	F[2] = 1 / ( 1 + (k*u[1])^2 ) - μ₂*u[2]

	return F
end

######################################################### targets and initial guess
targetData = StateDensity( 0:0.01:8, Ref(SizedVector{2}(4.0,5.0)) )
parameters = ( θ=SizedVector{3}(0.1,0.1,0.5), p=minimum(targetData.parameter))
u₀ = SizedVector{2}( [ SizedVector{2}(0.0,0.0) ], [ SizedVector{2}(0.0,0.0) ] )
