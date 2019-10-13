using Revise
include("inference.jl")

# parametrised hypothesis
function rates( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return Array([ p + θ₁*u[1] + θ₂*u[1]^3 + θ₃, u[2]^2 ])
end

# target state density
parameter = -2:0.05:2
density = ones(length(parameter)).*(abs.(parameter).<0.5)

# run inference
θ = param(randn(3))
infer( (u,p)->rates(u,p,θ...), θ, StateDensity(parameter,density); iter=200)

# visualising loss landscape
# using Flux.Tracker: update!
# global u₀,U,P
#
# θ = param([-2,1,-2])
# f = (u,p)->rates(u,p,θ...)
#
# data = StateDensity(parameter,density)
# maxSteps=1000
#
# # setting initial hyperparameters
# u₀ = param([-2.0])
# p₀ = param(minimum(parameter))
# pMax,ds = maximum(parameter), step(parameter)
# u₀,P,U = continuation( f,u₀,p₀; pMin=p₀-ds, pMax=pMax, ds=ds, maxSteps=maxSteps )
#
# function predictor()
# 	global u₀,U,P
# 	try
# 		u₀,P,U = continuation( f,u₀,p₀; pMin=p₀-ds, pMax=pMax, ds=ds, maxSteps=maxSteps )
# 		kernel = kde(P,data.parameter,bandwidth=1.4*ds) # density as label
# 		return kernel.density
# 	catch
# 		return NaN
# 	end
# end
#
# loss() = norm( predictor() .- data.density )
#
# function progress()
# 	@printf("Loss = %f, θ = %f,%f,%f\n", loss(), θ.data...)
# 	plot( P.data,U.data,
# 		label="inferred", color="darkblue",linewidth=3)
# 	plot!( data.parameter,predictor().data,
# 		label="inferred", color="darkblue")
# 	plot!( data.parameter,data.density,
# 		label="target", color="gold",
# 		xlabel="parameter, p", ylabel="steady state") |> display
# end
#
#
# function ℒoss(a,b)
# 	copyto!(θ.data,[a,b,0.0])
# 	return Tracker.data(loss())
# end
#
#
# θ = param([-1,-1,0])
# u₀,P,U = continuation( f,u₀,p₀; pMin=p₀-ds, pMax=pMax, ds=ds, maxSteps=maxSteps )
# x = y = range(-1,1,length=40)
# contourf(x,y, (x,y) -> ℒoss(x,y), size=(500,500))
