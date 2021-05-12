using Plots,FiniteDifferences
using LinearAlgebra: norm
using StatsBase: median

function grid_test(name; xlim=(-5,5),ylim=(-5,5),ϵ=-1e-1,order=5,threshold=0.05)

	hyperparameters = getParameters(targetData)
	parameters = copy(θ)

	function ∇cost(x::Number,y::Number)
		parameters[1:2] .= x,y

		L,∇L = ∇loss(rates,parameters,targetData,hyperparameters)
		return ∇L[1:2]
	end

	cost(x::Vector{<:Number}) = cost(x[1],x[2])
	function cost(x::Number,y::Number)
		parameters[1:2] .= x,y

		L = loss(rates,parameters,targetData,hyperparameters)
		return L
	end

	plot(size=(600,600), xlim=xlim, ylim=ylim, xlabel="parameters, θ")
	x,y = range(xlim...,length=50), range(ylim...,length=50)
	contour!( x, y, cost, alpha=0.5 )

	x,y = range(xlim...,length=12), range(ylim...,length=12)
	grid = collect(Iterators.product(x[2:end-1],y[2:end-1]))

	xGrid = vcat(map(x->x[1], grid)...)
	yGrid = vcat(map(x->x[2], grid)...)

	differences = map( (x,y)->first(grad(central_fdm(order,1,geom=true,condition=100), cost, [x,y] )), xGrid, yGrid )
	quiver!( xGrid, yGrid, quiver=(ϵ*step(x)*first.(differences),ϵ*step(y)*last.(differences)),
		color=:darkblue, lw=3 )

	gradients = map( ∇cost, xGrid, yGrid )
	quiver!( xGrid, yGrid, quiver=(ϵ*step(x)*first.(gradients),ϵ*step(y)*last.(gradients)),
		color=:gold, lw=2 )

	errors = norm.(differences .- gradients) ./ norm.(differences)
	errors = errors[.~isnan.(errors)]

	plot!([],[],color=:gold, lw=3, label="ForwardDiff", title="Median Error $(100*round(median(errors),digits=2))%")
	plot!([],[],color=:darkblue, lw=3, label="Central Differences")
	savefig(name)
end