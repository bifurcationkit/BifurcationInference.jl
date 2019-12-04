include("_inference.jl")
using Plots: Animation,frame,gif

######################################################## model
function rates( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return [ p + θ₁*u[1] + θ₂*u[1]^3 + θ₃ ]
end

function rates_jacobian( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return [ θ₁ .+ 3 .*θ₂.*u[1].^2 ][:,:]
end

# target state density
parameter = -2:0.1:2
data = StateDensity(parameter,[0.5,-0.5])

######################################################## run inference

parameters = ContinuationPar{Float64, typeof(DefaultLS()), typeof(DefaultEig())}(
	pMin=minimum(data.parameter),pMax=maximum(data.parameter),ds=step(data.parameter), maxSteps=1000,

		newtonOptions = NewtonPar{Float64, typeof(DefaultLS()), typeof(DefaultEig())}(
		verbose=false,maxIter=1000,tol= 1e-5),

	computeEigenValues = false)

u₀,θ,trace = [0.0],[2.1,2.1],[]
f,J = (u,p)->rates(u,p,θ...), (u,p)->rates_jacobian(u,p,θ...)

x,y = range(-4,4,length=100),range(-4,4,length=100)
contourf(x,y, (x,y) -> lossAt(x,y), size=(500,500))

function loss()
    bifurcations = predictor()

    if length(bifurcations.bifpoint) == 0
        return exp(-kurtosis( bifurcations.branch[1,:] ))

    elseif length(bifurcations.bifpoint) == 2
        return norm( data.bifurcations .- map( point -> point.param, bifurcations.bifpoint) )

	else
		return throw("unhandled bifurcations occured!")
    end
end

########################################################
u₀,θ,trace = [0.0],[2.1,2.1],[]
infer( f,J,u₀,θ, data; iter=250, progress = () -> push!(trace,copy(θ)) )
plot!( map( point -> point[1], trace), map( point -> point[2], trace),
	linewidth=1,color="white",markershape=:circle,markerstrokewidth=0,markersize=3,label="")

########################################################
u₀,θ,trace = [0.0],[-1.0,-1.0],[]
infer( f,J,u₀,θ, data; iter=250, progress = () -> push!(trace,copy(θ)) )
plot!( map( point -> point[1], trace), map( point -> point[2], trace),
	linewidth=1,color="white",markershape=:circle,markerstrokewidth=0,markersize=3,label="")

########################################################
u₀,θ,trace = [0.0],[3.0,-0.5],[]
infer( f,J,u₀,θ, data; iter=1500, progress = () -> push!(trace,copy(θ)) )
plot!( map( point -> point[1], trace), map( point -> point[2], trace),
	linewidth=1,color="white",markershape=:circle,markerstrokewidth=0,markersize=3,label="")


function progress()
    bifurcations = predictor()
    prediction = kde( bifurcations.branch[1,:], data.parameter, bandwidth=2*parameters.ds)

    plotBranch( bifurcations, label="inferred")
    plot!( prediction.x, prediction.density, label="inferred", color="darkblue")
    plot!( data.bifurcations, zeros(length(data.bifurcations)), linewidth=3, label="target", color="gold", xlabel="parameter, p", ylabel="steady state")
	frame(animation)
end

data = StateDensity(parameter,[1.5,0.5])
animation = Animation()
u₀,θ = [0.0],[-2.1,-2.1,0.0]
infer( f,J,u₀,θ, data; iter=250, progress=progress )
gif(animation,"examples/saddle-node-2.gif",fps=15)
