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

Ns,Ms = 1:10,1:5:50
complexity = scaling.(Ns,Ms')

contourf( Ns,Ms, complexity,
	xlabel="States", ylabel="Parameters", size=(500,400),
	colorbar_title="Iteration Execution / sec"
)

cticks!
savefig("test/scaling/scaling.pdf")

####################################################################################################
####################################################################################################
################################################################################ two-state-optima
using FluxContinuation,Plots,StaticArrays
using UMAP,Clustering,JLD
using StatsBase: median,mean

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
clusters = dbscan(optima,0.3,min_cluster_size=500)

layout = @layout [ a{0.7w} [b{0.5h}; b{0.5h} ] ]
default(); default(msc=:auto,label="",xlim=(-3,3),ylim=(-1,3),size=(600,450),grid=false)

fig = scatter([0],[0],markersize=0,title=L"\mathrm{Parameter\,\ Estimates}\,\,\theta^{*}", layout=layout, subplot=1,
	ylabel=L"\mathrm{Half\,\,\,Saturation}\quad\log_{10}k",xlabel=L"\mathrm{Activation}\quad \log_{10}a_1")

color = [:blue,:pink,:lightblue]
name = ["1","2","1′"]
clustered_names = ["1,1′","2"]
for (j,cluster) ∈ enumerate(clusters)
	x,y = getindex(optima,3,cluster.core_indices), getindex(optima,5,cluster.core_indices)

	scatter!(x,y,color=color[j],subplot=1)
	annotate!(median(x),median(y),name[j],subplot=1)
	
	parameters = getindex(optima,:,cluster.core_indices)
	i = rand(1:length(cluster.core_indices))
	if j == 1 X = StateSpace( 2, 0:0.002:10, [4,5] ) else X = StateSpace( 2, 0:0.01:10, [4,5] ) end
	if j ∈ [1,2]
		plot!(F,parameters[:,i],X,determinant=false,xlim=(0,10),ylim=(0.01,100),yscale=:log10,
			subplot=j+1, framestyle = :box)
		annotate!(8,30,clustered_names[j],subplot=j+1)
	end
end

xlabel!("",subplot=2)
ylabel!("",subplot=2)
ylabel!("",subplot=3)
yticks!([NaN],subplot=2)
yticks!([NaN],subplot=3)

savefig("docs/figures/two-state-optima.pdf")