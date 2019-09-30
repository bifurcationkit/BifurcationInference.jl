# Logistic growth model
using CSV, DataFrames, StatsPlots, Flux, DiffEqFlux

df = CSV.read("R33S175_Y81C76_OD_proc141021.txt")
#colnames = string.(names(data))
conditions = [replace(c, r"OD\(([^)]+)\)" => s"\1") for c in string.(names(df)[2:end])]

# Simulate the model multiple times
tdata = df[:,1]
nt, n = size(df[:,2:end])
data = [df[:,i+1] for i = 1:n]
x0 = 0.002
r = 0.1*randn(n) .+ 1.0
K = 0.2*randn(n) .+ 2.0
od0 = 0.1
x = zeros(nt,n)

function logistic_growth(du, u, p, t)
  r, K = p
  du[1] = r*u[1]*(1.0 - u[1]/K)
end
x0 = 0.002
tspan = (0.0,20)
p0 = [0.8, 2.3]
prob = ODEProblem(logistic_growth,[x0],tspan,p0)
p = param([r;K;log(x0);od0])
xs = solve(prob)
x = Flux.data.([Flux.data.(diffeq_rd([p[i],p[i+1]],prob,Tsit5(),saveat=tdata,u0=[exp(p[2*n+1])])[1,:]) for i = 1:n])

function inc!(A, b)
    for ii = 1:size(A, 1)
        for jj = 1:size(A[ii], 1)
            A[ii][jj] += b
        end
   end
end
inc!(x, od0)
x

# Produce a plot
function compare(sim, data)
    ph1 = scatter(tdata, data[1:12], xlim=[0,20], ylim=[0,3], markersize=1, layout=(3,4), legend=false)
    plot!(tdata, sim[1:12], layout=(3,4))

    ph2 = scatter(tdata, data[13:24], xlim=[0,20], ylim=[0,3], markersize=1, layout=(3,4), legend=false)
    plot!(tdata, sim[13:24], layout=(3,4))
    plot(ph1, ph2, layout=(2,1), size=(600,600))
end
compare(x, data)

function fit(r0, K0, x0, od0)
    p0 = [r0;K0;log(x0);od0]
    initialGradient(p0)
    p = param(p0)
    function predict()
        x = [diffeq_rd([p[i],p[i+n]],prob,Tsit5(),saveat=tdata,u0=[exp(p[2*n+1])])[1,:] for i = 1:n]
        #x = [diffeq_fd([p[i],p[i+n]],sol->sol[1,:],nt,prob,Tsit5(),saveat=tdata,u0=[exp(p[2*n+1])]) for i = 1:n]
        inc!(x, p[2*n+2])
        x
    end
    lossOD() = sum(map((xi,di) -> sum(abs2, xi - di), predict(), data))
    epochs = Iterators.repeated((), 200)
    opt = ADAM(0.2)
    cb = function ()
        #display(lossOD())
        @printf("loss = %1.3f (x0 = %1.4f, od0 = %1.3f)\n", Flux.data(lossOD()), exp(p[2*n+1]), p[2*n+2])
        x = predict()
        sim = Flux.data.(map(xi->Flux.data.(xi),x))
        display(compare(sim, data))
    end
    #cb()

    @time Flux.train!(lossOD, Flux.Params([p]), epochs, opt, cb=Flux.throttle(cb,0.2))
    #@time Flux.train!(lossOD, Flux.Params([p]), epochs, opt)
    cb()
    Flux.data.(p)
end

#pOpt = Flux.data.(p)
function initialGradient(p0)
    p = param(p0)
    function predict(p)
        x = [diffeq_rd([p[i],p[i+n]],prob,Tsit5(),saveat=tdata,u0=[exp(p[2*n+1])])[1,:] for i = 1:n]
        #x = [diffeq_fd([p[i],p[i+n]],sol->sol[1,:],nt,prob,Tsit5(),saveat=tdata,u0=[exp(p[2*n+1])]) for i = 1:n]
        inc!(x, p[2*n+2])
        x
    end
    lossOD(p) = sum(map((xi,di) -> sum(abs2, xi - di), predict(p), data))
    #display(p0)
    display(Tracker.gradient(lossOD,p0))
end
pOpt = fit(ones(n), 2*ones(n), 0.002, 0.1)

function plotParameters(p)
    r = Flux.data(p[1:24])
    K = Flux.data(p[25:48])
    #x0Opt = pOpt[25]
    rh = bar(1:n, r, ylabel="r", xticks = (1:n, conditions), xrotation=45, legend=false)
    Kh = bar(1:n, K, ylabel="K", xticks = (1:n, conditions), xrotation=45, legend=false)
    plot(rh, Kh, layout=(2,1), size=(600,600))
end
plotParameters(pOpt)
