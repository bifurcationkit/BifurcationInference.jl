include("inference.jl")

# parametrised hypothesis
function rates( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return [ p + θ₁*u[1] + θ₂*u[1]^3 + θ₃, -u[2] ]
end

function jacobian( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return [[ θ₁ .+ 3 .*θ₂.*u[1].^2, 0.0 ] [ 0.0, -1.0 ]]
end

# target state density
parameter = -2:0.05:2
density = ones(length(parameter)).*(abs.(parameter).<0.5)

######################################################## run inference
θ = param(randn(3))
f = (u,p)->rates(u,p,θ...)
J = (u,p)->jacobian(u,p,θ...)

infer( f, J, θ, StateDensity(parameter,density); iter=200)


######################################################## visualising loss landscape
using Flux.Tracker: update!
global u₀,U,P

# setting initial hyperparameters
data = StateDensity(parameter,density)
maxSteps=1000
maxIter=1000

P,U = param([0.0]),param([0.0])
u₀,p₀ = param([-2.0,0.0]),param(minimum(data.parameter))
pMax,ds = maximum(data.parameter), step(data.parameter)

function predictor()
	global u₀,U,P

	try u₀,P,U = continuation( f,J,u₀,p₀;
			pMin=p₀-ds, pMax=pMax, ds=ds,
			maxSteps=maxSteps, maxIter=maxIter )

	catch AssertionError end

	kernel = kde(P,data.parameter,bandwidth=1.4*ds)
	return kernel.density
end

loss() = norm( predictor() .- data.density )

function progress()
	@printf("Loss = %f, θ = [%f,%f,%f], u₀ = [%f,%f]\n", loss(), θ.data...,u₀...)
	plot( P.data,U.data,
		label="inferred", color="darkblue",linewidth=3)
	plot!( data.parameter,predictor().data,
		label="inferred", color="darkblue")
	plot!( data.parameter,data.density,
		label="target", color="gold",
		xlabel="parameter, p", ylabel="steady state") |> display
end


function ℒoss(a,b)
	copyto!(θ.data,[a,b,0.0])
	return Tracker.data(loss())
end


θ = param([-1,-1,0])
x,y = range(-1,1,length=60),range(-1,1,length=40)
contourf(x,y, (x,y) -> ℒoss(x,y), size=(500,500))
