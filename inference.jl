using Flux, Plots, KernelDensity, LinearAlgebra, Printf
include("continuation.jl")

struct StateDensity
	parameter::AbstractRange
	density::AbstractArray
end


function infer( f::Function, θ::TrackedArray, data::StateDensity; optimiser=ADAM(0.1),
		iter::Int=100, u₀::AbstractFloat=-2.0, uRange::AbstractFloat=1e3 )

	# setting initial hyperparameters
	p₀,pMax,ds = minimum(data.parameter), maximum(data.parameter), step(data.parameter)
	u₀,_,_= tangent( (u,p)->f(u,p).data ,u₀,p₀; ds=ds)
	U,P,∂ₚU = continuation( (u,p)->f(u,p).data ,u₀,p₀; ds=ds, pMax=pMax, uRange=uRange )

	function predictor()

		# predict parameter curve
		U,P,∂ₚU = continuation( f,u₀,p₀; ds=ds, pMax=pMax, uRange=uRange )
		u₀,_,_= tangent( (u,p)->f(u,p).data ,u₀,p₀; ds=ds)

		# state density as multi-stability label
		kernel = kde(P,data.parameter,bandwidth=1.05*ds)
		return kernel.density
	end

	function loss()
		density = predictor()
		boundary_condition = norm(∂ₚU[1]) + norm(∂ₚU[end])
		return norm(density.-data.density) + boundary_condition
	end

	function progress()
		@printf("Loss = %f, θ = %f,%f,%f\n", loss(), θ.data...)
		plot( P.data,U.data,
			label="inferred", color="darkblue",linewidth=3)
		plot!( data.parameter,data.density,
			label="target", color="gold",
			xlabel="parameter, p", ylabel="steady state") |> display
	end

	@time Flux.train!(loss, Flux.Params([θ]),
		Iterators.repeated((),iter), optimiser;
		cb=Flux.throttle(progress,5.0))
end


function rates( u, p=0.0, θ₂=0.0, θ₃=-1.0, θ₀=0.0 )
	return p + θ₂*u + θ₃*u^3 + θ₀
end

parameter = -2:0.05:2
density = ones(length(parameter)).*(abs.(parameter.+1.0).<0.2)

θ = param(randn(3))
infer( (u,p)->rates(u,p,θ...)/(norm(rates(u,p,θ...))+1), θ,
	StateDensity(parameter,density); iter=1000)


using Flux.Tracker: update!
θ = param([-2,1,-2])
global u₀,U,P
f = (u,p)->rates(u,p,θ...)/(norm(rates(u,p,θ...))+1)
data = StateDensity(parameter,density)
u₀,uRange=2.0,1e3

# setting initial hyperparameters
p₀,pMax,ds = minimum(data.parameter), maximum(data.parameter), step(data.parameter)
u₀,_,_= tangent( (u,p)->f(u,p).data ,u₀,p₀; ds=ds)
U,P,∂ₚU = continuation( (u,p)->f(u,p).data ,u₀,p₀; ds=ds, pMax=pMax, uRange=uRange )

function predictor()

	# predict parameter curve
	global u₀,U,P
	U,P,∂ₚU = continuation( f,u₀,p₀; ds=ds, pMax=pMax, uRange=uRange )
	u₀,_,_= tangent( (u,p)->f(u,p).data ,u₀,p₀; ds=ds)

	# state density as multi-stability label
	kernel = kde(P,data.parameter,bandwidth=1.05*ds)
	return kernel.density
end

function loss()
	density = predictor()
	boundary_condition = norm(∂ₚU[1]) + norm(∂ₚU[end])
	return norm(density.-data.density) + boundary_condition
end




function ℒoss(a,b)
	copyto!(θ.data,[a,1.0,b])
	@printf("a = %f, b = %f\n", a,b)
	return loss().data
end

ℒoss(-0.2,2)

x = y = range(-2,2,length=10)
contour(x,y, (x,y) -> ℒoss(x,y))
