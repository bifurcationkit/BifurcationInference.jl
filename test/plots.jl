using Plots
function grid_test(name; xlim=(-5,5),ylim=(-5,5), dims=[1,2], ϵ=-1e-1,order=5,geom=true,condition=100)

	hyperparameters = getParameters(X)
	parameters = copy(θ)

	function ∇cost(x::Number,y::Number)
		parameters[dims] .= x,y
		return ∇loss(F,parameters,X,hyperparameters)[2]
	end

	function cost(x::Number,y::Number)
		parameters[dims] .= x,y
		return loss(F,parameters,X,hyperparameters)
	end

	function finite_differences(x,y)
		parameters[dims] .= x,y
		return finite_difference_gradient(θ->loss(F,θ,X),parameters)
	end

	plot(size=(600,600), xlim=xlim, ylim=ylim, xlabel="parameters, θ")
	x,y = range(xlim...,length=50), range(ylim...,length=50)
	contour!( x, y, cost, alpha=0.5 )

	x,y = range(xlim...,length=12), range(ylim...,length=12)
	grid = collect(Iterators.product(x[2:end-1],y[2:end-1]))

	xGrid = vcat(map(x->x[1], grid)...)
	yGrid = vcat(map(x->x[2], grid)...)

	differences = map( finite_differences, xGrid, yGrid )
	quiver!( xGrid, yGrid, quiver=(ϵ*step(x)*getindex.(differences,dims[1]),ϵ*step(y)*getindex.(differences,dims[2])),
		color=:darkblue, lw=3 )

	gradients = map( ∇cost, xGrid, yGrid )
	quiver!( xGrid, yGrid, quiver=(ϵ*step(x)*getindex.(gradients,dims[1]),ϵ*step(y)*getindex.(gradients,dims[2])),
		color=:gold, lw=2 )

	errors = filter(x->~isnan(x), map( (x,y) -> acos(x'y/(norm(x)*norm(y)))/π, differences, gradients ))
	plot!([],[],color=:gold, lw=3, label="ForwardDiff", title="Mean Error $(100*round(sum(errors)/length(errors),digits=2))%")
	plot!([],[],color=:darkblue, lw=3, label="Central Differences")
	savefig(name)
end