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

f,J = (u,p)->rates(u,p,θ...), (u,p)->jacobian(u,p,θ...)
maxSteps,maxIter = 1000,1000
u₀, θ = param.([0]), param(randn(3))

infer( f,J,u₀,θ, data; iter=200, maxSteps=maxSteps, maxIter=maxIter)


######################################################## show loss

x,y = range(-1,1,length=40),range(-1,1,length=40)
contourf(x,y, (x,y) -> lossAt(x,y,0), size=(500,500))