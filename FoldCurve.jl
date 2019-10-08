using Flux, CuArrays, Plots, NLsolve, LinearAlgebra, StatsBase, Printf
using Flux.Tracker: gradient, update!
CuArrays.allowscalar(true)

# kernel density estimator on the gpu
rbf(x) = exp.(-x.^2/2)/sqrt(2π)
kde(x,data;width=0.05) = mean(rbf((x.-data')/width),dims=2)/width

function initial_tangent( rates, u₀, p₀ ; kwargs...)


	u,p,dp = u₀,p₀,kwargs[:dp]
	P = p+dp

	constraint = (p₀,u,p) -> p-p₀
	u,p = nlsolve( z -> ( rates(z...), constraint(p,z...) ), [u,p] ).zero
	U,P = nlsolve( z -> ( rates(z...), constraint(P,z...) ), [u,P] ).zero

	if u < u₀

		constraint = (u₀,u,p) -> u-u₀
		u,p = nlsolve( z -> ( rates(z...), constraint(u₀,z...) ), [u,p] ).zero
		U,P = nlsolve( z -> ( rates(z...), constraint(u₀,z...) ), [u,P] ).zero

	end

	∂ₚu = (U-u) / dp
	return u,p, ∂ₚu
end

function steady_state( rates, u₀, p₀ ; kwargs...)

	# initial tangent from simulation
	u₀,p₀, ∂ₚu = initial_tangent( rates, u₀, p₀ ; kwargs... )
	ds = kwargs[:dp]

	# psuedo-arclength constraint
	∂ₛu,∂ₛp = ∂ₚu,1.0
	constraint = (u₀,p₀,u,p) -> (u-u₀)*∂ₛu + (p-p₀)*∂ₛp - ds

	# main continuation loop
	U,P = [],[]
	while (p₀ < kwargs[:pMax]) & (u₀ < kwargs[:uMax])

		# predictor
		u,p = u₀ + ∂ₛu * ds, p₀ + ∂ₛp * ds

		# corrector
		u,p = nlsolve( z -> ( rates(z...), constraint(u₀,p₀,z...) ), [u,p] ).zero

		# update
		∂ₛu,∂ₛp = ( u - u₀ ) / ds, ( p - p₀ ) / ds
		u₀,p₀ = u,p

		# store
		push!(U,u); push!(P,p)
	end

	return Tracker.collect(U),Tracker.collect(P)
end

function rates( u, θ₁=0.0, θ₂=0.0, θ₃=-1.0, θ₀=0.0 )
	return θ₁ + θ₂*u + θ₃*u^3 + θ₀
end


function infer(u,p,θ; iter=100)
	u,p,θ = param(u),param(p),param(θ)
	Ugrid,Pgrid = collect(-2:0.005:2) ,collect(-2:0.01:2)

	target(u) = u.^3-u.+1.0
	data = kde(Pgrid,target(Ugrid))

	predictor(u,p) = steady_state(
		(u,p) -> rates(u,p,θ...),
		u,p; dp=0.01, pMax=2.0, uMax=2.0 )
	loss(state_density) = mean(state_density.*log.(state_density./data))

	function progress(u,p,state_density)
		@printf("Loss = %f, θ = %f,%f,%f\n", loss(state_density), θ.data...)

		plot( p.data,u.data, label="inferred", color="darkblue",linewidth=3)
		plot!( target(Ugrid), Ugrid, label="target", color="gold",linewidth=2,
			xlabel="parameter", ylabel="steady state", xlim=(-2.1,2.1), ylim=(-2.1,2.1))

		plot!( Pgrid,state_density.data, label="inferred", color="darkblue", linestyle=:dash )
		display(plot!( Pgrid,data, label="target", color="gold", linestyle=:dash ))
	end

	@time train!(loss, predictor, u,p,[θ], ADAM(0.1);
		iter=iter, progress=Flux.throttle(progress, 0.2))
end

function train!(loss, predictor, u,p,θ, optimiser;
	iter=100, progress = () -> ())

	Ugrid,Pgrid = collect(-2:0.005:2) ,collect(-2:0.01:2)
	θ = Flux.Params(θ)

	@progress for _ in Iterators.repeated((),iter)

		U,P = predictor(u,p)
		state_density = kde(Pgrid,P)
		progress(U,P,state_density)

		∂θ = gradient(θ) do
			loss(state_density) end

		update!(optimiser, θ, ∂θ)
	end
end

u,p = -2.0,-2.0
θ = [0.0,-2.0,1.0]
infer(u,p,θ; iter=200)
