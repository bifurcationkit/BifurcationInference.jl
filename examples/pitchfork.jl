include("_inference.jl")
using Plots: Animation,frame,gif

######################################################## model
function rates( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return [ θ₁ + p*u[1] + θ₂*u[1]^3 + θ₃, u[1]-u[2] ]
end

function rates_jacobian( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return [ [ p .+ 3 .*θ₂.*u[1].^2, 1.0] [0.0,-1.0] ]
end

function curvature( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return - 2u.^2 .* ( p .+ 3θ₂.*u.^2 ) .* ( 9*θ₂.*(p.+θ₂.*u.^2) .+ 2) ./ ( p.^2 .+ 6θ₂.*p.*u.^2 .+ 9θ₂^2 .*u.^4 .+ 2u.^2 ).^2
end

# initialise targets, model and hyperparameters
f = (u,p)->rates(u,p,θ...)
J = (u,p)->rates_jacobian(u,p,θ...)
K = (u,p)->curvature(u,p,θ...)

data = StateDensity(-5:0.01:5,[0.0])
parameters = getParameters(data)

######################################################## run inference
u₀,θ = [[0.0 0.0], [0.0 0.0], [0.0 0.0]], [1.0,0.5]

x = 6π/4
	lossAt(4cos(x),4sin(x))
	progress()

ϕ = range(0.03,2π-0.03,length=200)
L = lossAt.(4cos.(ϕ),4sin.(ϕ))

Lpoints = [L L]
Lrange = [zero(L).-3 L]

plot([],[NaN],fillrange=[NaN],
	title=L"\mathrm{Objective\,\,Function\,\,\,}L(\alpha)",
	label="",grid=false,proj=:polar,lims=(-3,2)
	)

	plot!(ϕ[1:10],Lpoints[1:10],fillrange=Lrange[1:10],
		color="lightblue",linewidth=3,fillalpha=0.4,label="")

	plot!(ϕ[11:29],Lpoints[11:29],fillrange=Lrange[11:29],
		color="#fde5d6",linewidth=3,fillalpha=0.3,label="")

	plot!(ϕ[30:70],Lpoints[30:70],fillrange=Lrange[30:70],
		color="lightblue",linewidth=3,fillalpha=0.3,label="")

	plot!(ϕ[71:89],Lpoints[71:89],fillrange=Lrange[71:89],
		color="#fde5d6",linewidth=3,fillalpha=0.3,label="")

	plot!(ϕ[90:100],Lpoints[90:100],fillrange=Lrange[90:100],
		color="lightblue",linewidth=3,fillalpha=0.4,label="")

	plot!(ϕ[101:111],Lpoints[101:111],fillrange=Lrange[101:111],
		color="darkblue",linewidth=3,fillalpha=0.2,label="")

	plot!(ϕ[112:130],Lpoints[112:130],fillrange=Lrange[112:130],
		color="#aabec7",linewidth=3,fillalpha=0.4,label="")

	plot!(ϕ[131:171],Lpoints[131:171],fillrange=Lrange[131:171],
		color="darkblue",linewidth=3,fillalpha=0.2,label="")

	plot!(ϕ[172:190],Lpoints[172:190],fillrange=Lrange[172:190],
		color="#aabec7",linewidth=3,fillalpha=0.4,label="")

	plot!(ϕ[191:200],Lpoints[191:200],fillrange=Lrange[191:200],
		color="darkblue",linewidth=3,fillalpha=0.2,label="")

	plot!(ϕ,zero(ϕ),color=:lightgray,label="")
	plot!(fill(x,2),[-3,-2.4], label=L"\mathrm{targets}\,\,\mathcal{D}",
		color="gold",linewidth=2)
savefig("objective-function-pitchfork.pdf")

# for bifurcation in predictor()
# 	bifpt = bifurcation.bifpoint[1:bifurcation.n_bifurcations]
#
# 	plot(bifurcation.branch[1,:],bifurcation.branch[2,:], alpha=0.5, label="", grid=false, frame=false, axis = nothing,
# 		color=map(x -> isodd(x) ? :darkblue : :lightblue, bifurcation.stability[1:bifurcation.n_points]), border=nothing, linewidth=20)
#
# 	scatter!(map(x->x.param,bifpt),map(x->x.printsol,bifpt), label="", border=nothing,
# 		m = (15.0, 15.0, :black, stroke(0, :none)), axis = nothing, frame=false) |> display
# end
# savefig("curve.pdf")

x = y = range(-5,5,length=50)
contourf( x, y, (x,y) -> lossAt(x,y) )

θ = Params([0.1,1.000])0
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
