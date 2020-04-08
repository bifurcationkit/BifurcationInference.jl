include("_inference.jl")
using Plots: Animation,frame,gif

######################################################## model
function rates( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return [ p + θ₁*u[1] + θ₂*u[1]^3 + θ₃ ]
end

function rates_jacobian( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return [ θ₁ .+ 3 .*θ₂.*u[1].^2 ][:,:]
end

function curvature( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return 6θ₂ .* ( 1 .+ θ₁^2 .- 9θ₂^2 .*u.^4 ) ./ (1 .+ (θ₁ .+ 3θ₂.*u.^2).^2 ).^2
end

# initialise targets, model and hyperparameters
f,J,K = (u,p)->rates(u,p,θ...), (u,p)->rates_jacobian(u,p,θ...), (u,p)->curvature(u,p,θ...)
data = StateDensity(-2:0.01:2,[0.5,-0.5])
parameters = getParameters(data)

######################################################## run inference

u₀,θ,A = [[0.0][:,:], [0.0][:,:] ],[2.1,-2.1], 1.0
progress()

plot(range(0,2π,length=200), ϕ->lossAt(3cos(ϕ),3sin(ϕ)))
vline!([14π/8-0.2],label="target", color="gold")

θ = [2.1,-2.1]
progress()
# x,y = range(-3,3,length=50),range(-7,7,length=50)
# contourf(x,y, (x,y) -> lossAt(x,y), size=(500,500))

θ = Params([0.5,-1.])
loss()

gs = gradient(θ) do
	loss()
end

import Base: iterate
iterate(::Nothing) = Nothing
iterate(::Nothing, x::Any) = x

########################################################
u₀,θ,trace = [0.0],[4.5,-1.0],[]
infer( f,J,u₀,θ, data; iter=100, progress = () -> push!(trace,copy(θ)) )
plot!( map( point -> point[1], trace), map( point -> point[2], trace),
	linewidth=1,color="white",markershape=:circle,markerstrokewidth=0,markersize=3,label="")

########################################################
u₀,θ,trace = [0.0],[-5.0,2.0],[]
infer( f,J,u₀,θ, data; iter=200, progress = () -> push!(trace,copy(θ)) )
plot!( map( point -> point[1], trace), map( point -> point[2], trace),
	linewidth=1,color="white",markershape=:circle,markerstrokewidth=0,markersize=3,label="")

########################################################
u₀,θ,trace = [0.0],[1.1,1.1],[]
infer( f,J,u₀,θ, data; iter=100, progress = () -> push!(trace,copy(θ)) )
plot!( map( point -> point[1], trace), map( point -> point[2], trace),
	linewidth=1,color="white",markershape=:circle,markerstrokewidth=0,markersize=3,label="")

########################################################
u₀,θ,trace = [0.0],[-1.0,-1.0],[]
infer( f,J,u₀,θ, data; iter=100, progress = () -> push!(trace,copy(θ)) )
plot!( map( point -> point[1], trace), map( point -> point[2], trace),
	linewidth=1,color="white",markershape=:circle,markerstrokewidth=0,markersize=3,label="")

########################################################
u₀,θ,trace = [0.0],[-3.0,-3.0],[]
infer( f,J,u₀,θ, data; iter=100, progress = () -> push!(trace,copy(θ)) )
plot!( map( point -> point[1], trace), map( point -> point[2], trace),
	linewidth=1,color="white",markershape=:circle,markerstrokewidth=0,markersize=3,label="")
