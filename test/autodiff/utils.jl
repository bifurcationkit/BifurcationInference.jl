using FiniteDifferences,BifurcationKit,Plots
global Iterable = BifurcationKit.ContIterable
using Setfield: @lens

######################################################## cost and gradient
function cost(θ::Vector{T}; ds::T=0.01) where T<:Number
	return sum( s->integrand(s.z,θ)*s.ds, solutions(θ,ds=ds) )
end

function ∇cost(θ::Vector{T}; ds::T=0.01) where T<:Number
	return sum( s->∇integrand(s.z,θ)*s.ds, solutions(θ,ds=ds) )
end

########################################################### utils
function solutions(θ::Vector{T}; ds::T=0.01, pMin::T=-5.0, pMax::T=5.0,
	maxSteps::Integer = 10000, maxIter::Integer = 1000, tol::T=1e-5 ) where T<:Number

	J(u,p) = ForwardDiff.jacobian(u->F(u,p,θ),u)
	solutions = NamedTuple{(:z, :ds),Tuple{Vector{T},T}}[]

	# parameters for the continuation
	newtonOptions = NewtonPar(tol = tol, verbose = false)
	options = ContinuationPar(dsmax = ds, dsmin = ds, ds = ds,

		maxSteps = maxSteps, pMin = pMin, pMax = pMax,
		saveSolEveryStep = 1, newtonOptions = newtonOptions )

	##################################### main continuation method
	v, _, converged, _ = newton( (u,p) -> F(u,p,θ), J, u, pMin, NewtonPar(newtonOptions; maxIter=maxIter))
	if converged u .= v else throw("newton not converged") end

	iterator = Iterable( (u,p)->F(u,p,θ), J, u, pMin, (@lens _), options, verbosity=0 )
	z′ = nothing

	for state ∈ iterator
		z = [state.z_old.u;state.z_old.p]

		push!( solutions, ( z = z, ds = isnothing(z′) ? NaN : norm(z-z′) ) )
		z′ = copy(z)
	end

	@assert(length(solutions)>3,"no solutions found for θ = $θ")
	solutions = solutions[2:end-1] # crop first and last points
	return solutions
end

∧(u::AbstractVector,v::AbstractVector) = u*v' - v*u'

function unit_test(; xlim=(-1,1),ylim=(-1,1),ϵ=5e-2,order=5)
	x,y = range(xlim...,length=30), range(ylim...,length=30)

	plot(size=(600,600), xlim=xlim, ylim=ylim, xlabel="parameters, θ")
	contour!( x, y, (x,y) -> cost([x,y]), alpha=0.5 )

	x,y = range(xlim...,length=12), range(ylim...,length=12)
	grid = collect(Iterators.product(x[2:end-1],y[2:end-1]))

	xGrid = vcat(map(x->x[1], grid)...) + 1e-3*randn(length(grid))
	yGrid = vcat(map(x->x[2], grid)...) + 1e-3*randn(length(grid))

	differences = map( (x,y)->first(grad(central_fdm(order,1,geom=true,condition=100), cost, [x,y] )), xGrid, yGrid )
	quiver!( xGrid, yGrid, quiver=(ϵ*first.(differences),ϵ*last.(differences)),
		color=:darkblue, lw=3 )

	gradients = map( (x,y)->∇cost([x,y]), xGrid, yGrid )
	quiver!( xGrid, yGrid, quiver=(ϵ*first.(gradients),ϵ*last.(gradients)),
		color=:gold, lw=2 )

	errors = norm.(differences .∧ gradients) ./ norm.(differences)
	errors = errors[.~isnan.(errors)]

	plot!([],[],color=:gold, lw=3, label="ForwardDiff", title="Mean Error $(round(100*sum(errors)/length(errors),digits=2))%")
	plot!([],[],color=:darkblue, lw=3, label="Central Differences") |> display
end

function plot_field(θ::Vector{T}; Δθ = 0.1, θidx = 1, Uidxs=1:2, ds=0.005, ϵ=1e-2) where T<:Number

	D = zeros(length(θ))
	D[θidx] = Δθ

	∂S,∂S₊,∇S = solutions(θ,ds=ds), solutions(θ+D,ds=ds), NamedTuple{(:z, :ds),Tuple{Vector{T},T}}[]
	for s ∈ ∂S push!( ∇S, ( z = s.z + Δθ*deformation(s.z,θ)[:,θidx], ds = s.ds ) ) end
	alpha,alpha₊,∇alpha = map(s->integrand(s.z,θ),∂S), map(s->integrand(s.z,θ),∂S₊), map(s->integrand(s.z,θ),∇S)

	@assert(maximum( z->maximum(deformation(z.z,θ)'tangent_field(z.z,θ)), ∂S)<1e-12)
	@assert(maximum( z->maximum(tangent_field(z.z,θ)'ForwardDiff.jacobian(θ->tangent_field(z.z,θ),θ)), ∂S)<1e-12)
	plot(size=(700,700),ylabel="states, u",xlabel="parameter, p",ylim=(-5,5),xlim=(-5,5))

	for Uidx ∈ Uidxs

		parameter,parameter₊,∇parameter = map(z->z.z[end],∂S), map(z->z.z[end],∂S₊), map(z->z.z[end],∇S)
		state,state₊,∇state = map(z->z.z[Uidx],∂S), map(z->z.z[Uidx],∂S₊), map(z->z.z[Uidx],∇S)
		
		plot!(parameter,  state,  label="", color=:darkblue,  alpha=alpha,  lw=3, linestyle=:solid)
		plot!(parameter₊, state₊, label="", color=:lightblue, alpha=alpha₊, lw=4, linestyle=:solid)
		plot!(∇parameter, ∇state, label="", color=:darkblue,  alpha=∇alpha, lw=1, linestyle=:solid)
	end

	plot!()|>display
end
