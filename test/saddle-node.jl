######################################################## model
function rates(u::Vector,θ::NamedTuple)
	@unpack p,z = θ; r,α,c = z
	θ₁,θ₂ = r*cos(α), r*sin(α)
	return [ p[1] + θ₁*u[1] + θ₂*u[1]^3 + c ]
end

function rates_jacobian(u::Vector,θ::NamedTuple)
	@unpack z = θ; r,α = z
	θ₁,θ₂ = r*cos(α), r*sin(α)
	return [ θ₁ + 3 *θ₂*u[1]^2 ][:,:]
end

function curvature(u::Vector,p::Number;θ::NamedTuple=θ)
	@unpack z = θ; r,α = z
	θ₁,θ₂ = r*cos(α), r*sin(α)
	return - 6θ₂ * ( 1 + θ₁^2 - 9θ₂^2 *u[1]^4 ) / (1 + (θ₁ + 3θ₂*u[1]^2)^2 )^2
end

######################################################### initialise targets, model and hyperparameters
targetData = StateDensity(-2:0.01:2,[1.0,-1.0])
hyperparameters = getParameters(targetData)

u₀ = [[0.0][:,:], [0.0][:,:]]
θ = ( z=[5.0,4.0,0.0], p=[minimum(targetData.parameter)])
