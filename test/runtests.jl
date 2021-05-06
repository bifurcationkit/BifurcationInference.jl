using FluxContinuation, StaticArrays
using Flux: Optimise

using StatsBase: median
using LinearAlgebra: norm

using FiniteDifferences
using Test

using Plots.PlotMeasures
using LaTeXStrings
using Plots

include("minimal/minimal.jl")

##################################################################
##################################################################
##################################################################

include("minimal/saddle-node.jl")
cost_landscape = unit_test("",xlim=(-5,5),ylim=(-5,5))
cost_landscape_copy = deepcopy(cost_landscape)

for θ₀ ∈ [ SizedVector{2}(1.5,1.5),SizedVector{2}(-4.0,0.1),SizedVector{2}(1.0,-2.5)]
    parameters = (θ=θ₀,p=-2.0)

    trajectory = train!(rates,parameters,targetData; iter=100, optimiser=Optimise.ADAM(0.1) )
    plot!( cost_landscape_copy, map(x->x[1],trajectory), map(x->x[2],trajectory),
        label="", lw=3, color=:gold )

    scatter!( cost_landscape_copy, [trajectory[end][1]], [trajectory[end][2]],
        label="", marker=:star5, color=:white, markersize=10 ) |> display

    plot(rates,trajectory[end],targetData) |> display
end

##################################################################
##################################################################
##################################################################

include("minimal/two-state.jl")
cost_landscape = unit_test("",xlim=(0.1,2.0),ylim=(0,4),ϵ=-1e-3)
cost_landscape_copy = deepcopy(cost_landscape)

##################################################################
##################################################################
##################################################################

trajectory = nothing
for (optimiser,iter,init,name) ∈ [
        (Optimise.ADAM(0.1),50,   (θ=SizedVector{2}(0.25,0.75),p=0.0),L"\mathrm{ADAM}\quad\eta = 0.01"),
        (Optimise.ADAM(0.1),50,   (θ=SizedVector{2}(0.5,3.0),p=0.0),L"\mathrm{ADAM}\quad\eta = 0.01"),
    ]

    parameters = init
    trajectory = train!(rates,parameters,targetData; iter=iter, noise=0, optimiser=optimiser )

    plot!( cost_landscape_copy,
        map(x->x[1],trajectory), map(x->x[2],trajectory),
        label="", lw=3, color=:gold )|> display
        
    scatter!( cost_landscape_copy, [trajectory[end][1]], [trajectory[end][2]],
        label="", marker=:star5, color=:white, markersize=10 )|> display
end

parameters = (θ=trajectory[end],p=0.0)
plot(rates,trajectory[end],targetData)

hyperparams = getParameters(targetData)
steady_states = deflationContinuation(rates,targetData.roots,parameters,hyperparams)

∇loss(rates,[0.6688847130664516, 1.4306067160382387],targetData,hyperparams)
∇loss(rates,[0.675750696058964, 1.4244707983433842],targetData,hyperparams)
∇loss(rates,[0.6819262843014389, 1.419049188686945],targetData,hyperparams)

∇loss(rates,[0.687478538045791, 1.4142742275532725],targetData,hyperparams)
∇loss(rates,[0.6924706930318885, 1.4100822983810475],targetData,hyperparams)
∇loss(rates,[0.6969571808565633, 1.4064182931355205],targetData,hyperparams)

∇loss(rates,SizedVector{2}(0.85,1.5),targetData,hyperparams)


parameters = (θ=SizedVector{2}(1.25,0.5),p=0.0)
trajectory = train!(rates,parameters,targetData; iter=25, optimiser=Descent(0.001) )

plot!( cost_landscape_copy, map(x->x[1],trajectory), map(x->x[2],trajectory),
    label="", lw=3, color=:white )

scatter!( cost_landscape_copy, [trajectory[end][1]], [trajectory[end][2]],
    label="", marker=:star5, color=:white, markersize=10 )

plot(rates,[0.8844708849993355, 0.9197039722333891],targetData)
    

parameters = (θ=SizedVector{2}(1.75,0.5),p=0.0)
trajectory = train!(rates,parameters,targetData; iter=500, optimiser=Descent(0.001) )

plot!( cost_landscape_copy, map(x->x[1],trajectory), map(x->x[2],trajectory),
    label="", lw=3, color=:white )

scatter!( cost_landscape_copy, [trajectory[end][1]], [trajectory[end][2]],
    label="", marker=:star5, color=:white, markersize=10 )


plot(rates,[1.3483722164851673, 1.7864847336030476],targetData)
    

parameters = (θ=SizedVector{2}(1.75,3.5),p=0.0)
trajectory = train!(rates,parameters,targetData; iter=500, optimiser=Descent(0.001) )

plot!( cost_landscape_copy, map(x->x[1],trajectory), map(x->x[2],trajectory),
    label="", lw=3, color=:white )

scatter!( cost_landscape_copy, [trajectory[end][1]], [trajectory[end][2]],
    label="", marker=:star5, color=:white, markersize=10 )

plot(rates,[1.4998761516914252, 2.4997792080399046],targetData)
    
##### two predictions one match
θ₀ = [0.87, 0.9]
θ₀ = [0.8, 1.0]

##### one prediction one match
θ₀ = [0.6, 1.1]
θ₀ = [0.6, 1.4]

##### two predictions one match
θ₀ = [1.0, 1.85]
θ₀ = [0.8, 1.3]

##### two predictions two match
θ₀ = [1.05, 1.53]

#### two predictions same match
θ₀ = [1.49, 2.0]
θ₀ = [1.49, 2.5]



θ₀ = [0.9, 1.2]
scatter!( cost_landscape_copy, [θ₀[1]], [θ₀[2]],
    label="", marker=:star5, color=:white, markersize=10 )

plot(rates,θ₀,targetData)


loss(rates,θ₀,targetData,hyperparams)
loss(rates,θ₀,targetData,hyperparams)