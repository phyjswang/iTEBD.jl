abstract type abstractiMPS{N, T<:Number} end

#-------------------------------------------------------------------------------------------
# BASIC PROPERTIES
#-------------------------------------------------------------------------------------------
get_data(mps::abstractiMPS{N, T}) where {N, T<:Number} = mps.Γ, mps.λ, mps.n
eltype(::abstractiMPS{N, T}) where {N, T<:Number} = T

#-------------------------------------------------------------------------------------------
function getindex(mps::abstractiMPS, i::Integer)
    i = mod(i-1, mps.n) + 1
    mps.Γ[i], mps.λ[i]
end

#-------------------------------------------------------------------------------------------
function setindex!(
    mps::abstractiMPS{N, T},
    v::Tuple{<:AbstractArray{<:Number, N}, <:AbstractVector{<:Real}},
    i::Integer
) where {N, T<:Number}
    i = mod(i-1, mps.n) + 1
    mps.Γ[i] = v[1]
    mps.λ[i] = v[2]
end

#-------------------------------------------------------------------------------------------
# MANIPULATION
#
# 1. conj       : Complex conjugation of iMPS.
# 4. entropy    : Return entanglement entropy across bond i.
#-------------------------------------------------------------------------------------------
export conj
function conj(mps::abstractiMPS)
    Γ, λ, n = get_data(mps)
    typeof(mps)(conj.(Γ), λ, n)
end

export normalize!
function normalize!(mps::abstractiMPS, i)
    ni = maximum(mps.λ[i])
    mps[i] = mps.Γ[i]/ni, mps.λ[i]/ni
    ni
end

#-------------------------------------------------------------------------------------------
export getEE
function getEE(mps::abstractiMPS, i::Integer)
    j = mod(i-1, mps.n) + 1
    ρ = mps.λ[j] .^ 2
    entanglement_entropy(ρ)
end

#-------------------------------------------------------------------------------------------
export getmaxD
function getmaxD(mps::abstractiMPS{N, T}) where {N, T<:Number}
    maximum([size(mps.λ[i],1) for i in 1:mps.n])
end
