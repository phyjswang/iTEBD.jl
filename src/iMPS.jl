#---------------------------------------------------------------------------------------------------
# iMPS
#---------------------------------------------------------------------------------------------------
export iMPS
"""
    iMPS

Infinite MPS.

Parameters:
- Γ : Vector of tensors.
- λ : Vector of Schmidt values.
- n : Number of tensors in the periodic blocks.

Note that the tensor `Γ` has absorbed the `λ` in, so it's in right canonical form.
"""
struct iMPS{T<:Number}
    Γ::Vector{Array{T, 3}}
    λ::Vector{Vector{Float64}}
    n::Int64
end
#---------------------------------------------------------------------------------------------------
function iMPS(
    T::DataType,
    Γs::AbstractVector{<:AbstractArray{<:Number, 3}}
)
    n = length(Γs)
    Γ = Array{T}.(Γs)
    λ = [ones(Float64, size(Γi, 3)) for Γi in Γs]
    iMPS(Γ, λ, n)
end
#---------------------------------------------------------------------------------------------------
function iMPS(Γs::AbstractVector{<:AbstractArray{<:Number, 3}})
    type_list = eltype.(Γs)
    T = promote_type(type_list...)
    iMPS(T, Γs)
end

#---------------------------------------------------------------------------------------------------
# BASIC PROPERTIES
#---------------------------------------------------------------------------------------------------
get_data(mps::iMPS) = mps.Γ, mps.λ, mps.n
eltype(::iMPS{T}) where T = T
#---------------------------------------------------------------------------------------------------
function getindex(mps::iMPS, i::Integer)
    i = mod(i-1, mps.n) + 1
    mps.Γ[i], mps.λ[i]
end
#---------------------------------------------------------------------------------------------------
function setindex!(
    mps::iMPS,
    v::Tuple{<:AbstractArray{<:Number, 3}, <:AbstractVector{<:Real}},
    i::Integer
)
    i = mod(i-1, mps.n) + 1
    mps.Γ[i] = v[1]
    mps.λ[i] = v[2]
end
#---------------------------------------------------------------------------------------------------
function mps_promote_type(
    T::DataType,
    mps::iMPS
)
    Γ, λ, n = get_data(mps)
    Γ_new = Array{T}.(Γ)
    iMPS(Γ_new, λ, n)
end
#---------------------------------------------------------------------------------------------------
function ent_S(mps::iMPS, i::Integer)
    j = mod(i-1, mps.n) + 1
    ρ = mps.λ[j] .^ 2
    entanglement_entropy(ρ)
end

#---------------------------------------------------------------------------------------------------
# INITIATE MPS
#
# 1. rand_iMPS    : Randomly generate iMPS with given bond dimension.
# 2. product_iMPS : Return iMPS from product state.
#---------------------------------------------------------------------------------------------------
export rand_iMPS
function rand_iMPS(
    T::DataType,
    n::Integer,
    d::Integer,
    dim::Integer
)
    Γ = [rand(T, dim, d, dim) for i=1:n]
    λ = [ones(dim) for i=1:n]
    iMPS(Γ, λ, n)
end
rand_iMPS(n, d, dim) = rand_iMPS(Float64, n, d, dim)
#---------------------------------------------------------------------------------------------------
export product_iMPS
function product_iMPS(
    T::DataType,
    v::AbstractVector{<:AbstractVector{<:Number}}
)
    n = length(v)
    d = length(v[1])
    Γ = [zeros(T, 1, d, 1) for i=1:n]
    λ = [ones(1) for i=1:n]
    for i=1:n
        Γ[i][1,:,1] .= v[i]
    end
    iMPS(Γ, λ, n)
end
function product_iMPS(v::AbstractVector{<:AbstractVector{<:Number}})
    T = promote_type(eltype.(v)...)
    product_iMPS(T, v)
end

#---------------------------------------------------------------------------------------------------
# MANIPULATION
#
# 1. conj       : Complex conjugation of iMPS.
# 2. applygate! : Apply gate to iMPS, return the result. The initial one will be altered.
# 3. gtrm       : Transfer matrix of the block periodic iMPS.
# 4. entropy    : Return entanglement entropy across bond i.
#---------------------------------------------------------------------------------------------------
export conj
function conj(mps::iMPS)
    Γ, λ, n = get_data(mps)
    iMPS(conj.(Γ), λ, n)
end
#---------------------------------------------------------------------------------------------------
# export applygate!
# function applygate!(
#     mps::iMPS,
#     G::AbstractMatrix{<:Number};
#     renormalize::Bool=false,
#     maxdim::Int64=MAXDIM,
#     cutoff::Float64=CUTOFF
# )
#     Γ, λ, n = get_data(mps)
#     Γ_new, λ_new = tensor_applygate!(G, Γ, λ[n], renormalize=renormalize, maxdim=maxdim, cutoff=cutoff)
#     Γ .= Γ_new
#     λ .= λ_new
#     mps
# end
# #---------------------------------------------------------------------------------------------------
# function applygate!(
#     mps::iMPS,
#     G::AbstractMatrix{<:Number},
#     inds::AbstractVector{<:Integer};
#     renormalize::Bool=false,
#     maxdim::Int64=MAXDIM,
#     cutoff::Float64=CUTOFF
# )
#     n = mps.n
#     indm = mod.(inds .- 1, n) .+ 1
#     indl = mod((inds[1]-2), n) + 1
#     Γs, λl = mps.Γ[indm], mps.λ[indl]
#     Γs_new, λs_new = tensor_applygate!(G, Γs, λl; renormalize, maxdim, cutoff)
#     mps.Γ[indm] .= Γs_new
#     mps.λ[indm] .= λs_new
#     mps
# end
#---------------------------------------------------------------------------------------------------
export gtrm
gtrm(mps1::iMPS, mps2::iMPS) = gtrm(mps1.Γ, mps2.Γ)
#---------------------------------------------------------------------------------------------------
export group
group(mps::iMPS) = tensor_group(mps.Γ)
#---------------------------------------------------------------------------------------------------
export decomposition!
function decomposition!(
    Γ::AbstractArray{<:Number, 3},
    λ::AbstractVector{<:Real},
    n::Integer;
    renormalize::Bool=false,
    maxdim::Integer=MAXDIM,
    cutoff::Real=CUTOFF
)
    tensor_lmul!(λ, Γ)
    Γs, λs = tensor_decomp!(Γ, λ, n, renormalize=renormalize, maxdim=maxdim, cutoff=cutoff)
    iMPS(Γs, λs, n)
end
#---------------------------------------------------------------------------------------------------
export transfer_matrix
transfer_matrix(mps::iMPS) = gtrm(mps, mps)

#---------------------------------------------------------------------------------------------------
# CANONICAL FORMS
#---------------------------------------------------------------------------------------------------
export canonical
function canonical(
    mps::iMPS;
    trim::Bool=false,
    renormalize::Bool=true,
    krylov_power::Integer=KRLOV_POWER,
    bound::Integer=BOUND,
    tol::Real=SVDTOL,
    zerotol::Real=ZEROTOL,
    sorttol::Real=SORTTOL,
)
    Γ_group = group(mps)
    if trim
        Γ_group = block_canonical(Γ_group, trim=true, krylov_power=krylov_power, zerotol=zerotol, sorttol=sorttol)
    end
    Γ_new, λ = schmidt_canonical(Γ_group, krylov_power=krylov_power, renormalize=renormalize, bound=bound, tol=tol, zerotol=zerotol)
    decomposition!(Γ_new, λ, mps.n, renormalize=renormalize, bound=bound, tol=tol)
end
