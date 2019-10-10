using DiffEqGPU, CuArrays, DifferentialEquations, Test

function lorenz(du,u,p,t)
 @inbounds begin
     du[1] = p[1]*(u[2]-u[1])
     du[2] = u[1]*(p[2]-u[3]) - u[2]
     du[3] = u[1]*u[2] - p[3]*u[3]
 end
 nothing
end

using GPUifyLoops, CuArrays, CUDAnative, DiffEqBase, LinearAlgebra

function gpu_kernel(f,du,u,p,t)
    @loop for i in (1:size(u,2); (blockIdx().x-1) * blockDim().x + threadIdx().x)
        @views @inbounds f(du[:,i],u[:,i],p[:,i],t)
        nothing
    end
    nothing
end

size([12 3 1])
gpu_kernel(A::CuArray) = @launch CUDA() kernel(A, threads=length(A))

data = CuArray{Float32}(undef, 1024)
kernel(data)

gpu_kernel(lorenz,0.,0.,0.)
