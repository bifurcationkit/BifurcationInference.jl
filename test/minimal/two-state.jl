######################################################## model
function rates(u::AbstractVector{T},parameters::NamedTuple) where T<:Number

	@unpack θ,p = parameters
	a₁,a₂,b₁,b₂,μ₁,μ₂,k = θ

	f = first(u)*first(p)*first(θ)
	F = similar(u,typeof(f))

	F[1] = ( a₁ + b₁ ) / ( 1 + (p*u[2])^2 ) - μ₁*u[1]
	F[2] = ( a₂ + b₂ ) / ( 1 + (k*u[1])^2 ) - μ₂*u[2]

	return F
end

######################################################### targets and initial guess
targetData = StateDensity(0:0.01:8,Ref([4.0,5.0]))
parameters = ( θ=[ 0.0,5.0, 7.0,0.0, 0.5,7.5, 0.5 ], p=minimum(targetData.parameter))
u₀ = [ [[0.0,0.0]], [[0.0,0.0]] ]