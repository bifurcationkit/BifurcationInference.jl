include("_inference.jl")

######################################################## model
function rates( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return [ p + θ₁*u[1] + θ₂*u[1]^3 + θ₃ ]
end

function jacobian( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return [ θ₁ .+ 3 .*θ₂.*u[1].^2 ][:,:]
end

# target state density
parameter = -2:0.1:2
density = ones(length(parameter)).*(abs.(parameter).<0.5)
data = StateDensity(parameter,density)

######################################################## run inference

parameters = ContinuationPar{Float64, typeof(DefaultLS()), typeof(DefaultEig())}(
	pMin=minimum(data.parameter),pMax=maximum(data.parameter),ds=step(data.parameter), maxSteps=1000,

		newtonOptions = NewtonPar{Float64, typeof(DefaultLS()), typeof(DefaultEig())}(
		verbose=false,maxIter=1000,tol= 1e-10),

	computeEigenValues = false)

f,J = (u,p)->rates(u,p,θ...), (u,p)->jacobian(u,p,θ...)
u₀, θ = [0.0], randn(3)

infer( f,J,u₀,θ, data; iter=200 )


######################################################## show loss
predictor()
gradient(loss,Params([θ]))

x,y = range(-1,1,length=50),range(-1,1,length=50)
contourf(x,y, (x,y) -> lossAt(x,y,0), size=(500,500))
