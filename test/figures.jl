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

hline!([0],subplot=2,linewidth=1,ylim=(-5,5),color=:black, xlabel=L"\mathrm{parameter,}p", xmirror=true, topmargin=-5mm,
	ylabel=L"\mathrm{determinant}\,\quad\left|\!\!\!\!\frac{\partial F_{\theta}}{\partial u}\right|")

plot!([-1.52,-1.52],[0,1],subplot=1,linewidth=2,color=:gold)
plot!([-1.52,-1.52],[0,5],subplot=2,linewidth=2,color=:gold)

plot!([1.33,1.33],[0,1],subplot=1,linewidth=2,color=:gold)
plot!([1.33,1.33],[0,5],subplot=2,linewidth=2,color=:gold)

plot!([1.67,1.67],[0,1],subplot=1,linewidth=2,color=:gold)
plot!([1.67,1.67],[0,5],subplot=2,linewidth=2,color=:gold)

p = -2:0.01:2
f(p) = - 2 - p^3 + p
plot!( p, measure, subplot=1, color=:gold, fill=true, alpha=0.2, ylabel=L"\mathrm{Measure}\,\,\varphi_{\theta}(z)", ylim=(0,1) )
plot!( p, f, subplot=2, linewidth=2, color=map( x -> f(x)<0 ? :red : :pink, p ) )

f(p) = - 2 - (p-1/2)^3 + 3.1(p-1/2)
plot!( p, measure, subplot=1, color=:gold, fill=true, alpha=0.2, ylim=(0,1) )
plot!( p, f, subplot=2, linewidth=2, color=map( x -> f(x)<0 ? :red : :pink, p ) )

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
heatmap( 1:5, 1:5, scaling,
	xlabel="States", ylabel="Parameters", size=(500,400),
	colorbar_title="Iteration Execution / sec"
)
savefig("test/scaling/scaling.pdf")

####################################################################################################
####################################################################################################
################################################################################ two-state-optima
using JLD,Plots
using UMAP,Clustering
using StatsBase: median

trajectories = Vector{Float64}[]
for file ∈ readdir(join=true,"trajectories")
	data = load(file)

	optima = map( (x,y)->(y<0.1)&(x>1),data["bifurcations"],data["losses"])
	append!(trajectories,data["trajectory"][optima])
end
optima = hcat(trajectories...)


embedding = umap(optima; n_neighbors=3, min_dist=2)
# clusters = dbscan(optima,0.1,min_neighbors = 1, min_cluster_size = 1)


scatter([0],[0],label="",markersize=0,size=(500,500),title="Reduced Parameter Space")

# for cluster ∈ clusters
	scatter!(getindex(optima,3,:),
		getindex(optima,4,:),msc=:auto,label="")
# end
scatter!([0],[0],label="",markersize=0)|>display

include("applied/two-state.jl")
for cluster ∈ clusters
	centroid = median(getindex(optima,:,cluster.core_indices),dims=2)[:,1]
	plot(rates,centroid,targetData)|>display
end
