include("inference.jl")

# parametrised hypothesis
function rates( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return [ p + θ₁*u[1] + θ₂*u[1]^3 + θ₃, u[1]-u[2] ]
end

function jacobian( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return [[ θ₁ .+ 3 .*θ₂.*u[1].^2, 1.0 ] [ 0.0, -1.0 ]]
end

# solve this and it can solve anytthing else
# function rates( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0, θ₄=0.0, θ₅=0.0, θ₆=0.0, θ₇=0.0, θ₈=0.0)
# 	return [ θ₁*p - ( θ₂ + θ₃*u[2] )*u[1],
# 			 θ₄*(1-u[2])/(0.01+1-u[2]) - (θ₆+θ₇*u[1])*u[2]/(0.01+u[2]) ]
# end
#
# function jacobian( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0, θ₄=0.0, θ₅=0.0, θ₆=0.0, θ₇=0.0, θ₈=0.0)
# 	return [[ θ₂ + θ₃*u[2] , θ₃*u[1] ] [ -θ₇*u[2]/(u[2]+0.01), -0.01*θ₄/(0.01+1-u[2])^2 - 0.01*(θ₆+θ₇*u[1])/(0.01+u[2])^2 ]]
# end

# target state density
parameter = -2:0.1:2
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

global u₀
p₀ = param(minimum(data.parameter))
pMax,ds = maximum(data.parameter), step(data.parameter)
u₀ = param.([-2,-2])

bifurcations, = continuation( f,J,u₀,p₀; pMin=p₀-ds, pMax=pMax, ds=ds,
	maxSteps=maxSteps, maxIter=maxIter, computeEigenValues=true )

prediction = kde( unpack(bifurcations)[1], data.parameter, bandwidth=1.4*ds)
u₀ = initial_state(bifurcations)

function predictor()
	global u₀

	bifurcations, = continuation( f,J,u₀, p₀; pMin=p₀-ds, pMax=pMax, ds=ds,
		maxSteps=maxSteps, maxIter=maxIter, computeEigenValues=true )

	density = kde( unpack(bifurcations)[1], data.parameter, bandwidth=1.4*ds).density
	u₀ = initial_state(bifurcations)

	return bifurcations,σ.((density.-0.5)./0.01)
end

function loss()
	try
		_,prediction = predictor()
		return norm( prediction .- data.density )
	catch
		return NaN end
end

function progress()
	@printf("Loss = %f, θ = [%f,%f,%f], u₀ = [%f,%f]\n", loss(), θ.data...,u₀...)
	bifurcations,prediction = predictor()
	plotBranch( bifurcations, label="inferred")
	plot!( data.parameter, prediction, label="inferred", color="darkblue")
	plot!( data.parameter,data.density, label="target", color="gold",
		xlabel="parameter, p", ylabel="steady state") |> display
end


function ℒoss(a,b)
	copyto!(θ.data,[a,b,0.0])
	return Tracker.data(loss())
end

ℒoss(-1,-1)

θ = param([-1,-1,0])
x,y = range(-1,1,length=60),range(-1,1,length=60)
contourf(x,y, (x,y) -> ℒoss(x,y), size=(500,500))
