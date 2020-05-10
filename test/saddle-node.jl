######################################################## model
function rates( u,p, α=0.0, r=5.0, c=0.0)
	θ₁,θ₂ = r*cos(α), r*sin(α)
	return [ p + θ₁*u[1] + θ₂*u[1]^3 + c ]
end

function rates_jacobian( u,p, α=0.0, r=5.0, c=0.0)
	θ₁,θ₂ = r*cos(α), r*sin(α)
	return [ θ₁ .+ 3 .*θ₂.*u[1].^2 ][:,:]
end

function curvature( u,p, α=0.0, r=5.0, c=0.0)
	θ₁,θ₂ = r*cos(α), r*sin(α)
	return - 6θ₂ .* ( 1 .+ θ₁^2 .- 9θ₂^2 .*u.^4 ) ./ (1 .+ (θ₁ .+ 3θ₂.*u.^2).^2 ).^2
end

######################################################### initialise targets, model and hyperparameters
f = (u,p)->rates(u,p,θ...)
J = (u,p)->rates_jacobian(u,p,θ...)
K = (u,p)->curvature(u,p,θ...)

data = StateDensity(-2:0.01:2,[1.0,-1.0])
parameters = getParameters(data)
u₀ = [[0.0][:,:], [0.0][:,:]]

optimum = 7π/4-0.14
ϕ,r = range(0,2π,length=10), 5.0
θ = [optimum,r]
