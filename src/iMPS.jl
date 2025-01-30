include("abstractiMPS.jl")

#-------------------------------------------------------------------------------------------
# iMPS
#-------------------------------------------------------------------------------------------
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
struct iMPS{T<:Number} <: abstractiMPS{3, T}
    Γ::Vector{Array{T, 3}}
    λ::Vector{Vector{Float64}}
    n::Int64
end

#-------------------------------------------------------------------------------------------
function iMPS(
    T::DataType,
    Γs::AbstractVector{<:AbstractArray{<:Number, 3}}
)
    n = length(Γs)
    Γ = Array{T}.(Γs)
    λ = [ones(Float64, size(Γi, 3)) for Γi in Γs]
    iMPS(Γ, λ, n)
end

#-------------------------------------------------------------------------------------------
function iMPS(Γs::AbstractVector{<:AbstractArray{<:Number, 3}})
    type_list = eltype.(Γs)
    T = promote_type(type_list...)
    iMPS(T, Γs)
end

#-------------------------------------------------------------------------------------------
export mps_promote_type
function mps_promote_type(
    T1::DataType,
    mps::iMPS{T}
) where T<:Number
    Γ, λ, n = get_data(mps)
    Γ_new = Array{T1}.(Γ)
    iMPS(Γ_new, λ, n)
end

#-------------------------------------------------------------------------------------------
# INITIATE MPS
#
# 1. rand_iMPS    : Randomly generate iMPS with given bond dimension.
# 2. product_iMPS : Return iMPS from product state.
#-------------------------------------------------------------------------------------------
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
#-------------------------------------------------------------------------------------------
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

#-------------------------------------------------------------------------------------------
# transfer matrix related
#-------------------------------------------------------------------------------------------
"""
 - ΓA - ΓB --...--
   |    |        [v]
 - ΓA - ΓB --...--
"""
export tmv_noconj
function tmv_noconj(mps::iMPS, v::Array{<:Number, 2})
    for i in mps.n:-1:1
        Γ = mps.Γ[i]
        @tensor v[:] := Γ[-1, 3, 1] * v[1, 2] * Γ[-2, 3, 2]
    end
    v
end
