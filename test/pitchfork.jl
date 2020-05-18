######################################################## model
function rates(u,θ)
	@unpack p, α, r, c = θ
	θ₁,θ₂ = r*cos(α), r*sin(α)
	return [ θ₁ + p*u[1] + θ₂*u[1]^3 + c, u[1]-u[2] ]
end

function rates_jacobian(u,θ)
	@unpack p, α, r = θ
	θ₁,θ₂ = r*cos(α), r*sin(α)
	return [ [ p + 3*θ₂*u[1]^2, 1.0] [0.0,-1.0] ]
end

function curvature(u,p;θ=θ)
	@unpack α, r = θ
	θ₁,θ₂ = r*cos(α), r*sin(α)
	return - 2u[1]^2 * ( p + 3θ₂*u[1]^2 ) * ( 9θ₂*(p+θ₂*u[1]^2) + 2) / ( p^2 + 6θ₂*p*u[1]^2 + 9θ₂^2 *u[1]^4 + 2u[1]^2 )^2
end

######################################################### initialise targets, model and hyperparameters
data = StateDensity(-5:0.01:5,[0.0])
hyperparameters = getParameters(data)

u₀ = [[0.0 0.0], [0.0 0.0], [0.0 0.0]]
θ = (r = 4.0, α = 6π/4+0.1, p=minimum(data.parameter), c = 0.0, α₀=6π/4)
ϕ = range(0.03,2π-0.03,length=100)
