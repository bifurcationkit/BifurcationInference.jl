######################################################## model
function rates(u::AbstractVector{T},parameters::NamedTuple) where T<:Number

	@unpack θ,p = parameters
	θ₁,θ₂ = θ[1]*cos(θ[2]), θ[1]*sin(θ[2])

	f = first(u)*first(p)*first(θ)
	F = similar(u,typeof(f))

	F[1] = θ₁ + p*u[1] + θ₂*u[1]^3
	F[2] = u[1] - u[2] # dummy second dimension

	return F
end

######################################################### targets and initial guess
targetData = StateDensity(-5:0.01:5,Ref([0.0]))
parameters = ( θ=[4.0,-π/2], p=minimum(targetData.parameter) )
u₀ = [ [[0.0,0.0]], [[0.0,0.0]] ]