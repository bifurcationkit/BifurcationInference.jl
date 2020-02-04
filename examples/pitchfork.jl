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
parameter = -2:0.01:2
data = StateDensity(parameter,[0.0])

######################################################## run inference
parameters = ContinuationPar{Float64, typeof(DefaultLS()), typeof(DefaultEig())}(
	pMin=minimum(data.parameter),pMax=maximum(data.parameter), maxSteps=100,
	ds=step(data.parameter), #dsmax=100*step(data.parameter), dsmin=step(data.parameter)/10,a=0.1,

		newtonOptions = NewtonPar{Float64, typeof(DefaultLS()), typeof(DefaultEig())}(
		verbose=false,maxIter=10,tol= 1e-12),

	detect_fold = false, detect_bifurcation = true)

f,J = (u,p)->rates(u,p,θ...), (u,p)->rates_jacobian(u,p,θ...)

θ = [1.0,0.5]
u₀ = [[0.0 0.0], [0.0 0.0], [0.0 0.0]]

x,y = range(-3,3,length=50),range(-3,3,length=50)
contourf(x,y, (x,y) -> lossAt(x,y), size=(500,500))

x = range(-3,3,length=500)
plot(x,x -> lossAt(x,-1.5))

θ = [-0.01,-1.1]
progress()

collect(x)[Int(end/2)]
θ = [1.0,0.06]
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
