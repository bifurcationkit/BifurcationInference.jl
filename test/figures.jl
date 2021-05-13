using LaTeXStrings,Plots
using Plots.Measures
using ForwardDiff

####################################################################################################
####################################################################################################
################################################################################ bifurcation measure

measure(x) = 1/(1+abs(f(x)/ForwardDiff.derivative(f,x)))
linewidth(x) = 3/(0.6-measure(x)/2)

plot(xlabel=L"\mathrm{parameter,}p", ylabel=L"\mathrm{determinant}\,\quad\left|\!\!\!\!\frac{\partial F_{\theta}}{\partial u}\right|",
	ylim=(-5,5), grid=false, right_margin=20mm, size=(500,450) )

hline!([0],linewidth=1,color=:black,label="")
p = -2:0.01:2

f(p) = - 2 - p^3 + p
plot!( p, f, label="", grid=false, color=:gold, alpha=0.1, linewidth=map(linewidth,p) )
plot!( p, f, label="", grid=false, linewidth=2, color=map( x -> f(x)<0 ? :red : :pink, p ) )

f(p) = - 2 - (p-1.5/4)^3 + 2.5(p-1.5/4)
plot!( p, f, label="", grid=false, color=:gold, alpha=0.3, linewidth=map(linewidth,p) )
plot!( p, f, label="", grid=false, linewidth=2, color=map( x -> f(x)<0 ? :red : :pink, p ) )

f(p) = - 2 - (p-1/2)^3 + 3.1(p-1/2)
plot!( p, f, label="", grid=false, color=:gold, alpha=0.5, linewidth=map(linewidth,p) )
plot!( p, f, label="", grid=false, linewidth=2, color=map( x -> f(x)<0 ? :red : :pink, p ) )

plot!( [NaN], [NaN], label=L"\mathrm{Bifuration\,Measure}\,\,\Psi_{\theta}(z)",
	grid=false, color=:gold, alpha=0.5, fill=true )
savefig("docs/figures/bifurcation-measure.pdf")