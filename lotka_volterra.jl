# Initial demonstration of ODE solving
using DifferentialEquations, Plots, Flux, DiffEqFlux

function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end

#u0_f(p,t0) = [p[5],p[6]]
u0 = [1.0, 1.0]
tspan = (0.0,10)
pOpt = [1.5,1.0,3.0,1.0]
prob = ODEProblem(lotka_volterra,u0,tspan,pOpt)
sol = solve(prob)
plot(sol)

# Add data
dt = 0.2
soldt = solve(remake(prob),Tsit5(),saveat=dt)
n = size(soldt[1,:])
data = soldt[1,:] + 0.05*randn(n).*soldt[1,:] + 0.25*randn(n)
t = soldt.t
scatter!(t, data, label="Data")

# Define function for optimizing Lotka-Volterra
sol1 = solve(remake(prob),Tsit5())

function fit(p0, data)
  p = param(p0)
  params = Flux.Params([p])

  predict_rd() = diffeq_rd(p,prob,Tsit5(),saveat=dt)[1,:]
    #diffeq_fd(p,sol->sol[1,:],51,prob,Tsit5(),saveat=dt)
    #diffeq_adjoint(p,prob,Tsit5(),saveat=dt)[1,:]
  loss_rd() = sum(abs2, predict_rd() - data)
  iter = Iterators.repeated((), 100)
  # Callback function to observe training
  function cb()
    display(loss_rd())
    plot(sol1, linewidth=1, linestyle=:dash, label=["Real u1", "Real u2"])
    scatter!(t, data, label="Noisy u1 (Data)")
    display(plot!(solve(remake(prob,p=Flux.data(p)),Tsit5())))
  end
  #@time Flux.train!(loss_rd, params, iter, ADAM(0.2), cb=Flux.throttle(cb,0.1))
  @time Flux.train!(loss_rd, params, iter, ADAM(0.2), cb=cb)
  #display(DataFrame(Parameters = ["α", "β", "δ", "γ"], Real = pOpt, Optimised = Flux.data(p)))
end

fit([1.0, 0.5, 2.0, 1.4], data)
