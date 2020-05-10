######################################################## model
function rates( u,p, α=0.0, r=4.0, c=0.0)
	θ₁,θ₂ = r*cos(α), r*sin(α)
	return [ θ₁ + p*u[1] + θ₂*u[1]^3 + c, u[1]-u[2] ]
end

function rates_jacobian( u,p, α=0.0, r=4.0, c=0.0)
	θ₁,θ₂ = r*cos(α), r*sin(α)
	return [ [ p .+ 3 .*θ₂.*u[1].^2, 1.0] [0.0,-1.0] ]
end

function curvature( u,p, α=0.0, r=4.0, c=0.0)
	θ₁,θ₂ = r*cos(α), r*sin(α)
	return - 2u.^2 .* ( p .+ 3θ₂.*u.^2 ) .* ( 9*θ₂.*(p.+θ₂.*u.^2) .+ 2) ./ ( p.^2 .+ 6θ₂.*p.*u.^2 .+ 9θ₂^2 .*u.^4 .+ 2u.^2 ).^2
end

# initialise targets, model and hyperparameters
f = (u,p)->rates(u,p,θ...)
J = (u,p)->rates_jacobian(u,p,θ...)
K = (u,p)->curvature(u,p,θ...)

data = StateDensity(-5:0.01:5,[0.0])
parameters = getParameters(data)
u₀ = [[0.0 0.0], [0.0 0.0], [0.0 0.0]]

optimum = 6π/4
ϕ,r = range(0.03,2π-0.03,length=200), 4.0
θ = [optimum,r]
