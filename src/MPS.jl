#---------------------------------------------------------------------------------------------------
# iMPS TYPE
#
# Parameters:
# Γ : Vector of tensors.
# λ : Vector of Schmidt values.
# n : Number of tensors in the periodic blocks.
#---------------------------------------------------------------------------------------------------
export iMPS
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
function iMPS(Γs::AbstractVector{<:AbstractArray{<:Number, 3}})
    type_list = eltype.(Γs)
    T = promote_type(type_list...)
    iMPS(T, Γs)
end

#---------------------------------------------------------------------------------------------------
# BASIC PROPERTIES
#---------------------------------------------------------------------------------------------------
export get_data
get_data(mps::iMPS) = mps.Γ, mps.λ, mps.n
eltype(::iMPS{T}) where T = T
function getindex(mps::iMPS, i::Integer)
    i = mod(i-1, mps.n) + 1
    mps.Γ[i], mps.λ[i]
end
function setindex!(
    mps::iMPS, 
    v::Tuple{<:AbstractArray{<:Number, 3}, <:AbstractVector{<:Real}},
    i::Integer
)
    i = mod(i-1, mps.n) + 1
    mps.Γ[i] = v[1]
    mps.λ[i] = v[2]
end
function mps_promote_type(
    T::DataType,
    mps::iMPS
)
    Γ, λ, n = get_data(mps)
    Γ_new = Array{T}.(Γ)
    iMPS(Γ_new, λ, n)
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
    λ = [ones(Float64, dim) for i=1:n]
    iMPS(Γ, λ, n)
end
rand_iMPS(n, d, dim) = rand_iMPS(Float64, n, d, dim)
#---------------------------------------------------------------------------------------------------
# MANIPULATION
#
# 1. conj       : Complex conjugation of iMPS.
# 2. applygate! : Apply gate to iMPS, return the result. The initial one will be altered.
# 3. gtrm       : Transfer matrix of the block periodic iMPS.
# 4. entropy    : Return entanglement entropy across bond i.
#---------------------------------------------------------------------------------------------------
function conj(mps::iMPS)
    Γ, λ, n = get_data(mps)
    iMPS(conj.(Γ), λ, n)
end
#---------------------------------------------------------------------------------------------------
export applygate!
function applygate!(
    mps::iMPS,
    G::AbstractMatrix{<:Number};
    renormalize::Bool=false,
    bound::Int64=BOUND,
    tol::Float64=SVDTOL
)
    Γ, λ, n = get_data(mps)
    Γ_new, λ_new = tensor_applygate!(G, Γ, λ[n], λ[n], renormalize=renormalize, bound=bound, tol=tol)
    Γ .= Γ_new
    λ .= λ_new
    mps
end
#---------------------------------------------------------------------------------------------------
function applygate!(
    mps::iMPS,
    G::AbstractMatrix{<:Number},
    inds::AbstractVector{<:Integer};
    renormalize::Bool=false,
    bound::Int64=BOUND,
    tol::Float64=SVDTOL
)
    n = mps.n
    indm = mod.(inds .- 1, n) .+ 1
    indl, indr = mod((inds[1]-2), n) + 1, indm[end]
    Γs, λl, λr = mps.Γ[indm], mps.λ[indl], mps.λ[indr]
    Γs_new, λs_new = tensor_applygate!(G, Γs, λl, λr, renormalize=renormalize, bound=bound, tol=tol)
    mps.Γ[indm] .= Γs_new
    mps.λ[indm] .= λs_new
    mps
end

#---------------------------------------------------------------------------------------------------
export transfer_matrix
gtrm(mps1::iMPS, mps2::iMPS) = gtrm(mps1.Γ, mps2.Γ)
transfer_matrix(mps::iMPS) = gtrm(mps, mps)
#---------------------------------------------------------------------------------------------------
export entropy
function entropy(
    mps::iMPS,
    i::Integer
)
    j = mod(i-1, mps.n) + 1
    ρ = mps.λ[j].^2
    entanglement_entropy(ρ)
end

#---------------------------------------------------------------------------------------------------
# CANONICAL FORMS
#---------------------------------------------------------------------------------------------------
export canonical
function canonical(
    mps::iMPS;
    trim::Bool=false,
    bound::Integer=BOUND,
    tol::Real=SVDTOL
)
    Γ, n = mps.Γ, mps.n
    Γ, λ = if trim
        canonical_trim(Γ, bound=bound, tol=tol)
    else
        schmidt_canonical(Γ, bound=bound, tol=tol)
    end
    iMPS(Γ, λ, n)
end
