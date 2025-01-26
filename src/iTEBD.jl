module iTEBD
#---------------------------------------------------------------------------------------------------
# CONSTANTS
#---------------------------------------------------------------------------------------------------
const MAXDIM = 50
const CUTOFF = 1e-8
const SORTTOL = 1e-3
const ZEROTOL = 1e-20
const KRLOV_POWER = 100
#---------------------------------------------------------------------------------------------------
# INCLUDE
#---------------------------------------------------------------------------------------------------
using LinearAlgebra
using SparseArrays
using TensorOperations
using KrylovKit
import Base: eltype, getindex, setindex!
import LinearAlgebra: conj, normalize!

include("abstractiMPS.jl")
include("iMPS.jl")
include("iMPO.jl")
include("Gate.jl")
include("TensorAlgebra.jl")
include("Schmidt.jl")
include("Block.jl")
include("Krylov.jl")
include("Miscellaneous.jl")

end
