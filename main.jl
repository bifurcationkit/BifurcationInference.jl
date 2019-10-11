include("inference.jl")

# parametrised hypothesis
function rates( u, p=0.0, θ₂=0.0, θ₃=-1.0, θ₀=0.0 )
	return p + θ₂*u + θ₃*u^3 + θ₀
end

# target state density
parameter = -2:0.05:2
density = ones(length(parameter)).*(abs.(parameter).<0.5)

# run inference
θ = param(randn(3))
infer( (u,p)->rates(u,p,θ...), θ,
	StateDensity(parameter,density); iter=1000)

using Flux.Tracker: update!
global u₀,U,P
θ = param([-2,1,-2])

f = (u,p)->rates(u,p,θ...)
data = StateDensity(parameter,density)
u₀,uRange=1.0,1e3

# setting initial hyperparameters
p₀,pMax,ds = minimum(data.parameter), maximum(data.parameter), step(data.parameter)
u₀,_,_= tangent( (u,p)->f(u,p).data ,u₀,p₀; ds=ds)
U,P = continuation( (u,p)->f(u,p).data ,u₀,p₀; ds=ds, pMax=pMax, uRange=uRange )

function predictor()

	# predict parameter curve
	global u₀,U,P
	U,P = continuation( f,u₀,p₀; ds=ds, pMax=pMax, uRange=uRange )
	u₀,_,_= tangent( (u,p)->f(u,p).data ,u₀,p₀; ds=ds)

	# state density as multi-stability label
	kernel = kde(P,data.parameter,bandwidth=1.05*ds)
	return kernel.density
end
loss() = norm( predictor() .- data.density) + 10*(norm(θ)-1)^2

function progress()
	@printf("Loss = %f, θ = %f,%f,%f\n", loss(), θ.data...)
	plot( P.data,U.data,
		label="inferred", color="darkblue",linewidth=3)
	plot!( data.parameter,predictor().data,
		label="inferred", color="darkblue")
	plot!( data.parameter,data.density,
		label="target", color="gold",
		xlabel="parameter, p", ylabel="steady state") |> display
end


function ℒoss(a,b)
	copyto!(θ.data,[a,b,0.0])
	return loss().data
end

θ = param([-1,-1,0])
u₀,_,_= tangent( (u,p)->f(u,p).data ,u₀,p₀; ds=ds)
x = range(-1,1,length=24)
y = range(-1,1,length=24)
contourf(x,y, (x,y) -> ℒoss(x,y), size=(500,500))

ℒoss(-1,1)
progress()
