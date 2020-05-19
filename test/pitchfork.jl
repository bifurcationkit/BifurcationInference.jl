######################################################## model
function rates(u::Vector,θ::NamedTuple)
	@unpack p,z = θ; r,α,c = z
	θ₁,θ₂ = r*cos(α), r*sin(α)
	return [ θ₁ + p[1]*u[1] + θ₂*u[1]^3 + c, u[1]-u[2] ]
end

function rates_jacobian(u::Vector,θ::NamedTuple)
	@unpack p,z = θ; r,α = z
	θ₁,θ₂ = r*cos(α), r*sin(α)
	return [ [ p[1] + 3*θ₂*u[1]^2, 1.0] [0.0,-1.0] ]
end

function curvature(u::Vector,p::Number;θ::NamedTuple=θ)
	@unpack z = θ; r,α = z
	θ₁,θ₂ = r*cos(α), r*sin(α)
	return - 2u[1]^2 * ( p + 3θ₂*u[1]^2 ) * ( 9θ₂*(p+θ₂*u[1]^2) + 2) / ( p^2 + 6θ₂*p*u[1]^2 + 9θ₂^2 *u[1]^4 + 2u[1]^2 )^2
end

######################################################### initialise targets, model and hyperparameters
targetData = StateDensity(-5:0.01:5,[0.0])
hyperparameters = getParameters(targetData)

u₀ = [[0.0 0.0], [0.0 0.0], [0.0 0.0]]
θ = ( z=[4.0,6π/4+1,0.0], p=[minimum(targetData.parameter)])
