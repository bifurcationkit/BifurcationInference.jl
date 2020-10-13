using StaticArrays: lu,StaticMatrix
import LinearAlgebra: factorize
factorize(A::StaticMatrix) = lu(A)
