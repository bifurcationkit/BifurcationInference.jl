# using PseudoArcLengthContinuation: ContinuationPar,NewtonPar,DefaultLS,DefaultEig,AbstractLinearSolver,AbstractEigenSolver
# using Flux,FluxContinuation
#
# function infer( f::Function, J::Function, u₀::Vector{T}, θ::AbstractArray, data::StateDensity;
#         optimiser=ADAM(0.05), progress::Function=()->(), iter::Int=100, maxIter::Int=10, tol=1e-12 ) where T
#
#     parameters = getParameters(data; maxIter=maxIter, tol=tol)
#     @time train!(loss, Params([θ]), iter, optimiser, cb=progress)
# end

# ############################################# determinant, trace and curvature
# function κ(branch)
# 	p,u,ds = branch.branch[1,:], branch.branch[2,:], branch.branch[4,:]
#     return K(u,p), abs.(ds)
# end

############################################# objective function
