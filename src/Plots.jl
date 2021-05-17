using Plots.PlotMeasures
using LaTeXStrings
using Plots

import Plots: plot
function plot(steady_states::Vector{<:Branch},data::StateSpace)

	layout = @layout [a;b{1.0w,0.5h}]
	default(); default(grid=false,label="",margin=1mm,linewidth=2)
	figure = plot(layout = layout, link = :x, size=(300,500) )

	hline!([0],subplot=1,linewidth=0,color=:black, ylabel=L"\mathrm{steady\,states}\quad F_{\theta}(z)=0")
	hline!([0],subplot=2,linewidth=1,color=:black, xlabel=L"\mathrm{parameter,}p", xmirror=true, topmargin=-5mm,
		ylabel=L"\mathrm{determinant}\,\quad\left|\!\!\!\!\frac{\partial F_{\theta}}{\partial u}\right|")
	
	vline!(data.targets,subplot=1,linewidth=1,color=:gold)
	vline!(data.targets,subplot=2,linewidth=1,color=:gold)

    for branch ∈ steady_states

        stability = map( s -> all(real(s.λ).<0), branch)
        determinants = map( s -> prod(real(s.λ)), branch)
		parameter = map( s -> s.z.p, branch)

		for idx ∈ 1:dim(branch)

			plot!( parameter, map(s->s.z.u[idx],branch), subplot=1,
				color=map( stable -> stable ? :darkblue : :lightblue, stability ) )
		end

		plot!( parameter, determinants, subplot=2,
            color=map( stable -> stable ? :red : :pink, stability )
		)
    end

	xticks!([NaN],subplot=1)
	return figure
end

function plot(F::Function, θ::AbstractVector, data::StateSpace; kwargs...)

	parameters = (θ=θ,p=minimum(data.parameter))
	hyperparameters = getParameters(data;kwargs...)

	steady_states = deflationContinuation(F,data.roots,parameters,hyperparameters;kwargs...)
	println(steady_states)

	return plot(steady_states,data)
end
