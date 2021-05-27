using FluxContinuation
using ForwardDiff

using LaTeXStrings,Plots
using Plots.Measures

####################################################################################################
####################################################################################################
######################################################################################## saddle-node
include("minimal/saddle-node.jl"); X.roots .= [[[-1.0]]]
plot(F,[5/2,-1],X)
savefig("docs/figures/saddle-node.pdf")

####################################################################################################
####################################################################################################
########################################################################################## pitchfork
include("minimal/pitchfork.jl")
plot(F,[1/2,-1],X)
savefig("docs/figures/pitchfork.pdf")

####################################################################################################
####################################################################################################
################################################################################ bifurcation measure
measure(x) = 1/(1+abs(f(x)/ForwardDiff.derivative(f,x)))

layout = @layout [a;b{1.0w,0.5h}]
default(); default(grid=false,label="",margin=0mm)
plot(layout = layout, link = :x, size=(400,500) )

hline!([0],subplot=2,linewidth=1,ylim=(-5,5),color=:black, xlabel=L"\mathrm{arclength,}s", xmirror=true, topmargin=-5mm,
	ylabel=L"\mathrm{determinant}\,\quad\left|\!\!\!\!\frac{\partial F_{\theta}}{\partial u}\right|")

plot!([-1.52,-1.52],[0,1],subplot=1,linewidth=2,color=:gray)
plot!([-1.52,-1.52],[0,5],subplot=2,linewidth=2,color=:gray)

plot!([1.33,1.33],[0,1],subplot=1,linewidth=2,color=:gray)
plot!([1.33,1.33],[0,5],subplot=2,linewidth=2,color=:gray)

plot!([1.67,1.67],[0,1],subplot=1,linewidth=2,color=:gray)
plot!([1.67,1.67],[0,5],subplot=2,linewidth=2,color=:gray)

p = -2:0.01:2
f(p) = - 2 - (p-1/2)^3 + 3.1(p-1/2)
plot!( p, measure, subplot=1, linewidth=2, color=:red, ylim=(0,1) )
plot!( p, f, subplot=2, linewidth=2, color=:red )

f(p) = - 2 - p^3 + p
plot!( p, measure, subplot=1, linewidth=2, color=:gold, ylabel=L"\mathrm{Measure}\,\,\varphi_{\theta}(s)", ylim=(0,1) )
plot!( p, f, subplot=2, linewidth=2, color=:gold )

yticks!([0,1],subplot=1)
yticks!([-2,0,2],subplot=2)
xticks!([NaN],subplot=1)

scatter!([1.67],[0],subplot=2,marker=:star,color=:black,markersize=7)
scatter!([1.33],[0],subplot=2,marker=:star,color=:black,markersize=7)
scatter!([-1.52],[0],subplot=2,marker=:circle,color=:black,markersize=4)


savefig("docs/figures/bifurcation-measure.pdf")

####################################################################################################
####################################################################################################
############################################################################################ scaling
include("scaling/scaling.jl")
using StatsBase: mean,std
using LaTeXStrings

complexity_mean = Float64[]
complexity_std = Float64[]

for n ∈	1:40
	ts = [scaling(n,3) for _ ∈ 1:10]
	push!(complexity_mean,mean(ts))
	push!(complexity_std,std(ts))
end
scaling(40,3)

default();default(grid=false)
plot( 10:30, x->x^3/1800, label=L"N^3", color=:red, linewidth=2, legend=:topleft)
scatter!( complexity_mean,yerror=complexity_std, label="", yscale=:log10, xscale=:log10, ms=3, color=:black,
	xlabel=L"\mathrm{States}\,\,N", ylabel=L"\mathrm{Execution\,\,Time}\,\,/\,\,\mathrm{sec}", size=(400,400) )

xticks!([1,3,5,10,20,30],["1","3","5","10","20","30"])
yticks!([1e-1,1e0,1e1],["0.1","1","10"])
savefig("docs/figures/scaling.pdf")

####################################################################################################
####################################################################################################
################################################################################ two-state-optima
using FluxContinuation,Plots,StaticArrays
using Clustering,JLD
using StatsBase: median

trajectories, losses, bifurcations, K = Vector{Float64}[], Float64[], Float64[], 0
for file ∈ readdir(join=true,"trajectories")
	data = load(file)

	mask = @. (data["bifurcations"]==2)&(data["losses"]<0.1)
	if ~any(x->any(isnan.(x)),data["trajectory"])

		append!(trajectories,data["trajectory"][mask])
		append!(bifurcations,data["bifurcations"][mask])
		append!(losses,data["losses"][mask])
		K += 1
	end
end

K/length(readdir("trajectories"))
optima = hcat(trajectories...)

@. optima[end,:] = abs(optima[end,:])
M,N = size(optima)
clusters = dbscan(optima,0.5,min_cluster_size=40)

layout = @layout [ a{0.7w} [b{0.5h}; b{0.5h} ] ]
default(); default(msc=:auto,label="",size=(600,450),grid=false)

fig = scatter([0],[0],markersize=0,title=L"\mathrm{Parameter\,\ Estimates}\,\,\theta^{*}", layout=layout, subplot=1,
	ylabel=L"\mathrm{Sensitivity/Degredation\,\,Ratio}\quad k/\mu_1",xlabel=L"\mathrm{Basal\,\,Production\,\,of}\,\,u_1\,,\quad a_1")

color = [:lightblue,:pink]
include("applied/two-state.jl")
for (j,cluster) ∈ enumerate(clusters)
	x,y = 10 .^(getindex(optima,3,cluster.core_indices)), getindex(optima,5,cluster.core_indices) ./ 10 .^getindex(optima,1,cluster.core_indices)
	@assert all( i -> sign(getindex(optima,3,i)) == sign(getindex(optima,4,i)), cluster.core_indices )

	scatter!(x,y,color=color[j],subplot=1,xscale=:log10,xlim=(1e-3,1e3),yscale=:log10,ylim=(1e-1,1e2))
	annotate!(median(x),median(y),j,subplot=1)
	
	parameters = getindex(optima,:,cluster.core_indices)
	if j == 1 X = StateSpace( 2, 0:0.002:10, [4,5] ) else X = StateSpace( 2, 0:0.01:10, [4,5] ) end

	plot!(F,parameters[:,rand(1:length(cluster.core_indices))],X,
		determinant=false,xlim=(0,10),ylim=(0.01,100),yscale=:log10,
		subplot=j+1, framestyle = :box)
	annotate!(8.8,30,j,subplot=j+1)
end

xlabel!("",subplot=2)
ylabel!("",subplot=2)
ylabel!("",subplot=3)
yticks!([NaN],subplot=2)
yticks!([NaN],subplot=3)

savefig("docs/figures/two-state-optima.pdf")