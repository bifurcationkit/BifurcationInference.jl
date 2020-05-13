############################################# hyperparameter updates
function getParameters(data::StateDensity{T}; maxIter::Int=10, tol=1e-12) where {T<:Number}
    return ContinuationPar(

        pMin=minimum(data.parameter),pMax=maximum(data.parameter), maxSteps=10*length(data.parameter),
        ds=step(data.parameter), dsmax=step(data.parameter), dsmin=step(data.parameter),

            newtonOptions = NewtonPar(
            verbose=false,maxIter=maxIter,tol=tol),

        detectFold = false, detectBifurcation = true)
end
@nograd getParameters

function updateParameters(parameters::ContinuationPar{T, S, E}, steady_states::Vector{Branch{T}};
    resolution=200 ) where {T<:Number, S<:AbstractLinearSolver, E<:AbstractEigenSolver}

    # estimate scale from steady state curves
    branch_points = map(length,steady_states)
    ds = maximum(branch_points)*parameters.ds/resolution
    parameters = setproperties(parameters;ds=ds,dsmin=ds,dsmax=ds)

    return parameters
end
@nograd updateParameters

############################################################## plotting
import Plots: plot
function plot(steady_states::Vector{Branch{T}}, data::StateDensity{T}; idx::Int=1) where {T<:Number}

    vline( data.bifurcations, label="", color=:gold, xlabel=L"\mathrm{parameter},\,p",
        right_margin=20mm,size=(500,400)); right_axis = twinx()

    for branch in steady_states

        plot!(branch.parameter, map(x->x[idx],branch.state), linewidth=2, alpha=0.5, label="", grid=false,
            ylabel=L"\mathrm{steady\,states}\quad F_{\theta}(z)=0",
            #color=map(x -> isodd(x) ? :darkblue : :lightblue, branch.stability )
            )

        # determinant = map( x -> ( (eigenvalues,vectors,i) = x; prod(eigenvalues) ), branch.eig)
        # plot!(right_axis, branch.parameter, determinant, linewidth=2, alpha=0.5, label="", grid=false,
        # 	ylabel=L"\mathrm{determinant}\,\,\Delta_{\theta}(z)",
        #     #colour=map(x -> isodd(x) ? :red : :pink, branch.stability )
        # 	)

        scatter!(map(x-> x.parameter, branch.bifurcations),
                 map(x-> x.state[idx],branch.bifurcations),
            label="", m = (3.0,3.0,:black,stroke(0,:none)))
    end

    plot!(right_axis,[],[], color=:gold, legend=:bottomleft,
        alpha=1.0, label=L"\mathrm{targets}\,\,\mathcal{D}")

        scatter!(right_axis,[],[], label=L"\mathrm{prediction}\,\,\mathcal{P}(\theta)", legend=:bottomleft,
            m = (1.0, 1.0, :black, stroke(0, :none)))
        plot!(right_axis,[],[], color=:darkblue, legend=:bottomleft, linewidth=2,
            alpha=1.0, label=L"\mathrm{steady\,states}")
        plot!(right_axis,[],[], color=:red, legend=:bottomleft,
            alpha=1.0, label=L"\mathrm{determinant}", dpi=500, linewidth=2) |> display
end
@nograd plot
