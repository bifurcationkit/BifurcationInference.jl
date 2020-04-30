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
	return - 6θ₂ .* ( 1 .+ θ₁^2 .- 9θ₂^2 .*u.^4 ) ./ (1 .+ (θ₁ .+ 3θ₂.*u.^2).^2 ).^2
end

# initialise targets, model and hyperparameters
f = (u,p)->rates(u,p,θ...)
J = (u,p)->rates_jacobian(u,p,θ...)
K = (u,p)->curvature(u,p,θ...)

data = StateDensity(-2:0.01:2,[1.0,-1.0])
parameters = getParameters(data)

######################################################## run inference
u₀,θ = [[0.0][:,:], [0.0][:,:] ],[2.1,-2.1]

x = 7π/4+0.11
	lossAt(3cos(x),3sin(x))
	progress()

ϕ = range(0,2π,length=200)
L = lossAt.(3cos.(ϕ),3sin.(ϕ))

Lpoints = [L L]
Lrange = [zero(L).-4 L]

plot([],[NaN],fillrange=[NaN],
	title=L"\mathrm{Objective\,\,Function\,\,\,}L(\alpha)",
	label="",grid=false,#proj=:polar,lims=(-4,3)
	)

	plot!(ϕ[1:50],Lpoints[1:50],fillrange=Lrange[1:50],
		color="#fde5d6",linewidth=3,fillalpha=0.3,label="")

	plot!(ϕ[51:91],Lpoints[51:91],fillrange=Lrange[51:91],
		color="lightblue",linewidth=3,fillalpha=0.4,label="")

	plot!(ϕ[92:150],Lpoints[92:150],fillrange=Lrange[92:150],
		color="#aabec7",linewidth=3,fillalpha=0.4,label="")

	plot!(ϕ[151:190],Lpoints[151:190],fillrange=Lrange[151:190],
		color="darkblue",linewidth=3,fillalpha=0.2,label="")

	plot!(ϕ[191:200],Lpoints[191:200,:],fillrange=Lrange[191:200,:],
		color="#fde5d6",linewidth=3,fillalpha=0.3,label="")

	plot!(ϕ,zero(ϕ),color=:lightgray,label="")
	plot!(fill(x,2),[-4,-3], label=L"\mathrm{targets}\,\,\mathcal{D}",
		color="gold",linewidth=2)
savefig("objective-function-saddle.pdf")
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
