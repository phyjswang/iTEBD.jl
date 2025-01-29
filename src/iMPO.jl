include("abstractiMPS.jl")

#-------------------------------------------------------------------------------------------
# iMPO
#-------------------------------------------------------------------------------------------
export iMPO
"""
    iMPO

Infinite MPO.

Parameters:
- Γ : Vector of tensors.
- λ : Vector of Schmidt values.
- n : Number of tensors in the periodic blocks.

Note that the tensor `Γ` has absorbed the `λ` in, so it's in right canonical form.
   |
   2
   |
-1-Γ-4-
   |
   3
   |
"""
struct iMPO{T<:Number} <: abstractiMPS{4, T}
    Γ::Vector{Array{T, 4}}
    λ::Vector{Vector{Float64}}
    n::Int64
end
#-------------------------------------------------------------------------------------------
function iMPO(
    T::DataType,
    Γs::AbstractVector{<:AbstractArray{<:Number, 4}}
)
    n = length(Γs)
    Γ = Array{T}.(Γs)
    λ = [ones(Float64, size(Γi, 4)) for Γi in Γs]
    iMPO(Γ, λ, n)
end
#-------------------------------------------------------------------------------------------
function iMPO(Γs::AbstractVector{<:AbstractArray{<:Number, 4}})
    type_list = eltype.(Γs)
    T = promote_type(type_list...)
    iMPO(T, Γs)
end

#-------------------------------------------------------------------------------------------
export mpo_promote_type
function mpo_promote_type(
    T1::DataType,
    mpo::iMPO{T}
) where T<:Number
    Γ, λ, n = get_data(mpo)
    Γ_new = Array{T1}.(Γ)
    iMPO(Γ_new, λ, n)
end


#-------------------------------------------------------------------------------------------
# INITIATE MPO
#
# 1. rand_iMPO    : Randomly generate iMPO with given bond dimension.
# 2. product_iMPO : Return iMPO from product state.
#-------------------------------------------------------------------------------------------
export rand_iMPO
function rand_iMPO(
    T::DataType,
    n::Integer,
    d::Integer,
    dim::Integer
)
    Γ = [rand(T, dim, d, d, dim) for i=1:n]
    λ = [ones(dim) for i=1:n]
    iMPO(Γ, λ, n)
end
rand_iMPO(n, d, dim) = rand_iMPO(Float64, n, d, dim)
#-------------------------------------------------------------------------------------------
export product_iMPO
function product_iMPO(
    T::DataType,
    v::AbstractVector{<:AbstractVector{<:Number}}
)
    n = length(v)
    d = length(v[1])
    Γ = [zeros(T, 1, d, d, 1) for i=1:n]
    λ = [ones(1) for i=1:n]
    for i=1:n
        Γ[i][1,:,:,1] .= v[i]
    end
    iMPO(Γ, λ, n)
end
function product_iMPO(v::AbstractVector{<:AbstractVector{<:Number}})
    T = promote_type(eltype.(v)...)
    product_iMPO(T, v)
end


#-------------------------------------------------------------------------------------------
# transfer matrix related
#-------------------------------------------------------------------------------------------
export tmv
"""
    ----     ----
   |    |   |    |
 --ΓA* -+-- ΓB* -+----
   |    |   |    |    [v]
 --+----ΓA--+----ΓB---
   |    |   |    |
    ----     ----
"""
function tmv(mps::iMPO, v::Array{<:Number, 2})
    for i in mps.n:-1:1
        Γ = mps.Γ[i]
        @tensor v[:] := (Γ[-1, 1, 2, 3] * v[3, 4]) * conj(Γ)[-2, 1, 2, 4]
    end
    v
end

"""
    -----     -----
   |     |   |     |
 --Γ0A* -+-- Γ0B* -+----
   |     |   |     |    [v]
 --+----Γ1A--+----Γ1B---
   |     |   |     |
    -----     -----
"""
function tmv(mps1::iMPO, mps0::iMPO, v::Array{<:Number, 2})
    for i in mps1.n:-1:1
        Γ1 = mps1.Γ[i]
        Γ0 = mps0.Γ[i]
        @tensor v[:] := (Γ1[-1, 1, 2, 3] * v[3, 4]) * conj(Γ0)[-2, 1, 2, 4]
    end
    v
end
