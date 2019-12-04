include("_inference.jl")
using Plots: Animation,frame,gif

######################################################## model
function rates( u, y₁=0.0, y₂=0.0, μ₁=0.0, μ₂=0.0, a₁=0.0, a₂=0.0, b₁=0.0, b₂=0.0, n=2)
	return [ a₁/(1+(y₁*u[4])^n) - μ₁*u[1],  b₁*u[1] - μ₂*u[2],
			 a₂/(1+(y₂*u[2])^n) - μ₁*u[3],  b₂*u[3] - μ₂*u[4] ]
end

function rates_jacobian( u, y₁=0.0, y₂=0.0, μ₁=0.0, μ₂=0.0, a₁=0.0, a₂=0.0, b₁=0.0, b₂=0.0, n=2)

	_2to3 = -n*u[2]^(n-1)*a₂*y₂^n/(1+(y₂*u[2])^n)^2
	_4to1 = -n*u[4]^(n-1)*a₁*y₁^n/(1+(y₁*u[4])^n)^2

	return [[ -μ₁,    b₁,  0.0,  0.0  ] [ 0.0,   -μ₂, _2to3, 0.0  ] [ 0.0,   0.0,   -μ₁,  b₂  ] [ _4to1, 0.0,   0.0, -μ₂  ]]
end

# target state density
parameter = 0:0.05:7
data = StateDensity(parameter,[5.0,4.0])

function progress()
    bifurcations = predictor()
    prediction = kde( bifurcations.branch[1,:], data.parameter, bandwidth=2*parameters.ds)

    plotBranch( bifurcations, label="inferred")
    plot!( prediction.x, prediction.density, label="inferred", color="darkblue")
    plot!( data.bifurcations, zeros(length(data.bifurcations)), linewidth=3, label="target", color="gold", xlabel="parameter, p", ylabel="steady state")
	frame(animation)
end

######################################################## run inference

parameters = ContinuationPar{Float64, typeof(DefaultLS()), typeof(DefaultEig())}(
	pMin=minimum(data.parameter),pMax=maximum(data.parameter),ds=step(data.parameter), maxSteps=1000,

		newtonOptions = NewtonPar{Float64, typeof(DefaultLS()), typeof(DefaultEig())}(
		verbose=false,maxIter=1000,tol= 1e-5),

	computeEigenValues = false)

f,J = (u,p)->rates(u,p,θ...), (u,p)->rates_jacobian(u,p,θ...)
u₀,θ = [4.0,3.0,0.0,0.0], [ 2.5, 0.5, 7.5, 4.0, 2.0, 0.4, 1.5 ]

animation = Animation()
infer( f,J,u₀,θ, data; iter=200, optimiser=Momentum(0.001,0.7),
	progress=progress )

gif(animation,"examples/toggle-switch.gif",fps=15)
