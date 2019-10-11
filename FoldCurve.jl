using Flux, Plots, KernelDensity, LinearAlgebra, Printf
using Flux.Tracker: gradient, update!
using Flux: binarycrossentropy,crossentropy
include("continuation.jl")

struct data
	parameter::AbstractArray
	density::AbstractArray
end

function rates( u, p=0.0, θ₂=0.0, θ₃=-1.0, θ₀=0.0 )
	return p + θ₂*u + θ₃*u^3 + θ₀
end

function infer( f,θ, data::data; iter=100, u₀=-2.0, uRange=1e3 )

	# setting initial hyperparameters
	ds = step(data.parameter)
	p₀,pMax = minimum(data.parameter), maximum(data.parameter)
	u₀,p₀,_= tangent( (u,p)->f(u,p).data ,u₀,p₀; ds=ds)

	function predictor()

		# predict parameter curve
		curve = continuation( f,u₀,p₀; ds=ds, pMax=pMax, uRange=uRange )
		u₀,_,_= tangent( (u,p)->f(u,p).data ,u₀,p₀; ds=ds)

		# compute multi-stability label
		kernel = kde(curve[2,:],data.parameter,bandwidth=1.05*ds)
		label = kernel.density
		return curve,label
	end

	function loss(density,∂ₚU)
		boundary_condition = norm(∂ₚU[1]) + norm(∂ₚU[end])
		return norm(density.-data.density) + boundary_condition
	end

	function progress(u,p,density,∂ₚU)
		@printf("Loss = %f, θ = %f,%f,%f\n", loss(density,∂ₚU), θ.data...)
		plot( p.data,u.data,
			label="inferred", color="darkblue",linewidth=3)
		plot!( data.parameter, density.data,
			label="inferred", color="darkblue", linestyle=:dash )
		plot!( data.parameter,data.density,
			label="target", color="gold", linestyle=:dash,
			xlabel="parameter, p", ylabel="steady state",
			xlim=(-2.1,2.1), ylim=(-2.1,2.1)) |> display
	end

	@time train!(loss, predictor, [θ], ADAM(0.1);
		iter=iter, progress=Flux.throttle(progress,5.0))
end

function train!(loss, predictor, θ, optimiser; iter=100, progress = () -> ())
	θ = Flux.Params(θ)

	@progress for _ in Iterators.repeated((),iter)

		output,state_density = predictor()
		U,P,∂ₚU = output[1,:],output[2,:],output[3,:]
		progress(U,P,state_density,∂ₚU)

		∂θ = gradient(θ) do
			loss(state_density,∂ₚU) end

		update!(optimiser, θ, ∂θ)
	end
end


parameter = -2:0.05:2
density = ones(length(parameter)).*(abs.(parameter.+1.0).<0.2)

θ = param(randn(3))
infer( (u,p)->rates(u,p,θ...)/(norm(rates(u,p,θ...))+1), θ,
	data(parameter,density); iter=1000)
