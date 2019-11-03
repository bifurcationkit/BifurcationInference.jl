include("_inference.jl")

######################################################## model
function rates( u, y₁=0.0, y₂=0.0, μ₁=0.0, μ₂=0.0, a₁=0.0, a₂=0.0, b₁=0.0, b₂=0.0, n=2)
	return [ a₁/(1+(y₁*u[4])^n) - μ₁*u[1],  b₁*u[1] - μ₂*u[2],
			 a₂/(1+(y₂*u[2])^n) - μ₁*u[3],  b₂*u[3] - μ₂*u[4] ]
end

function jacobian( u, y₁=0.0, y₂=0.0, μ₁=0.0, μ₂=0.0, a₁=0.0, a₂=0.0, b₁=0.0, b₂=0.0, n=2)

	_2to3 = -n*u[2]^(n-1)*a₂*y₂^n/(1+(y₂*u[2])^n)^2
	_4to1 = -n*u[4]^(n-1)*a₁*y₁^n/(1+(y₁*u[4])^n)^2

	return [[ -μ₁,    b₁,  0.0,  0.0  ] [ 0.0,   -μ₂, _2to3, 0.0  ] [ 0.0,   0.0,   -μ₁,  b₂  ] [ _4to1, 0.0,   0.0, -μ₂  ]]
end

# target state density
parameter = 0:0.1:10
density = ones(length(parameter)).*(abs.(parameter.-4).<0.5)
data = StateDensity(parameter,density)

######################################################## run inference

maxSteps,maxIter = 1000,1000
tol = 1e-5

loss()
progress()

f,J = (u,p)->rates(u,p,θ...), (u,p)->jacobian(u,p,θ...)
u₀,θ = param.([4,3,0,0]), param([ 2.5, 0.5, 0.5, 2.0, 2.0, 0.4, 0.4 ])

infer( f,J,u₀,θ, data; iter=200, maxSteps=maxSteps, maxIter=maxIter, tol=tol)





Tracker.gradient( ()-> loss() )

######################################################## show loss

x,y = range(-1,1,length=40),range(-1,1,length=40)
contourf(x,y, (x,y) -> lossAt(x,y,0), size=(500,500))
