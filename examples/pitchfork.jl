include("_inference.jl")
using Plots: Animation,frame,gif

######################################################## model
function rates( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return [ θ₁ + p*u[1] + θ₂*u[1]^3 + θ₃, u[1]-u[2] ]
end

function rates_jacobian( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return [ [ p .+ 3 .*θ₂.*u[1].^2, 1.0] [0.0,-1.0] ]
end

# initialise targets, model and hyperparameters
f,J = (u,p)->rates(u,p,θ...), (u,p)->rates_jacobian(u,p,θ...)
data = StateDensity(-2:0.01:2,[0.0])
parameters = getParameters(data)

θ = [0.12,-0.5]
progress()
png("pitchfork")

######################################################## run inference

u₀,θ,A = [[0.0 0.0], [0.0 0.0], [0.0 0.0]], [1.0,0.5], 1.0
plot(range(0,2π,length=100), ϕ->lossAt(3cos(ϕ),3sin(ϕ)))
vline!([2π/3+0.05],label="target", color="gold")

# x,y = range(-3,3,length=50),range(-3,3,length=50)
# contourf(x,y, (x,y) -> lossAt(x,y), size=(500,500))



θ = Params([0.1,1.000])
loss()

gs = gradient(θ) do
	loss()
end


# x = range(-3,3,length=500)
# plot(x,x -> lossAt(-1.5,x))
#
# θ = [-1.5,-0.518]
# progress()
# u₀
#
# predictor()
# collect(x)[Int(end/2)]
# θ = [1.0,0.06]
# progress()
#
#


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
