using Flux, Plots, NLsolve, LinearAlgebra
using Flux.Tracker: gradient, update!

function initial_tangent( rates, u₀, p₀ ; kwargs...)

	constraint = (p₀,u,p) -> p-p₀
	u,p,dp = u₀,p₀,kwargs[:dp]
	P = p+dp

	u,p = nlsolve( z -> ( rates(z...), constraint(p,z...) ), [u,p] ).zero
	U,P = nlsolve( z -> ( rates(z...), constraint(P,z...) ), [u,P] ).zero

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
	while p₀ < kwargs[:pMax]

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

function rates( u, θ₁=0.0, θ₂=0.0, θ₃=-1.0 )
	return θ₁ + θ₂*u + θ₃*u^3
end


function fit(u,p,θ; iter=100)
	u,p,θ = param(u),param(p),param(θ)

	predictor(u,p) = steady_state(
		(u,p) -> rates(u,p,θ...),
		u,p; dp=0.01, pMax=2.0 )
	target(u) = u.^3-u
	loss(u,p) = norm(p-target(u))

	function progress(u,p)
		loss(u,p) |> display
		plot( Flux.data(p),Flux.data(u),
			label="inferred", color="darkblue",linewidth=3)
		plot!( Flux.data(target(u)), Flux.data(u),
			label="target", color="gold",linewidth=2,
			xlabel="parameter", ylabel="steady state",
			xlim=(-2,2), ylim=(-2,2)) |> display
	end

	@time train!(loss, predictor, u,p,[θ], ADAM(0.1); iter=iter, progress=progress)
end

function train!(loss, predictor, u,p,θ, optimiser; iter=100, progress = () -> ())
	θ = Flux.Params(θ)
	@progress for _ in Iterators.repeated((),iter)

		U,P = predictor(u,p)
		∂θ = gradient(θ) do
			loss(U,P) end

		update!(optimiser, θ, ∂θ)
		progress(U,P)
	end
end

u,p = -1.0,-2.0
θ = [1.0,-3.0]
fit(u,p,θ; iter=100)












function Π(x,width=1.0,ϵ=1e-3)
	return σ.((x .+ width/2)/ϵ) .* σ.((width/2 .- x)/ϵ)
end

function ρ( A; dx=0.01 )

	xmin,xmax = minimum(A),maximum(A)
	x = range(xmin.data,xmax.data,step=dx)

	ρₓ = map( x -> sum(Π(A.-x,dx)), x)
	ρₓ ./= sum(ρₓ)*dx

	return x,Tracker.collect(ρₓ)
end
