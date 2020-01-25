include("_inference.jl")
using Plots: Animation,frame,gif

######################################################## model
function rates( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return [ θ₁ + p*u[1] + θ₂*u[1]^3 + θ₃, u[1]-u[2] ]
end

function rates_jacobian( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return [ [ p .+ 3 .*θ₂.*u[1].^2, 1.0] [0.0,-1.0] ]
end

# target state density
parameter = -2:0.1:2
data = StateDensity(parameter,[0.05,-0.05])

######################################################## run inference
parameters = ContinuationPar{Float64, typeof(DefaultLS()), typeof(DefaultEig())}(
	pMin=minimum(data.parameter),pMax=maximum(data.parameter),ds=step(data.parameter), maxSteps=1000,

		newtonOptions = NewtonPar{Float64, typeof(DefaultLS()), typeof(DefaultEig())}(
		verbose=false,maxIter=5000,tol= 1e-12),

	detect_fold = false, detect_bifurcation = true)

u₀,θ,trace = [0.0,0.0],[2.1,2.1],[]
f,J = (u,p)->rates(u,p,θ...), (u,p)->rates_jacobian(u,p,θ...)

x,y = range(-3,3,length=50),range(-3,3,length=50)
plot(x, x-> lossAt(x,2.0) )
contourf(x,y, (x,y) -> lossAt(x,y), size=(500,500))

θ = [x[end],2.0]
progress()

θ = Params([0.1,1.000])
loss()

gs = gradient(θ) do
	loss()
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
