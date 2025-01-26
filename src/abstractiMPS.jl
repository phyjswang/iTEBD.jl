abstract type abstractiMPS{N, T} end

#-------------------------------------------------------------------------------------------
# BASIC PROPERTIES
#-------------------------------------------------------------------------------------------
get_data(mps::abstractiMPS) = mps.Γ, mps.λ, mps.n
eltype(::abstractiMPS{N, T}) where {N, T} = T

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
) where {N, T}
    i = mod(i-1, mps.n) + 1
    mps.Γ[i] = v[1]
    mps.λ[i] = v[2]
end

#-------------------------------------------------------------------------------------------
function mps_promote_type(
    T::DataType,
    mps::abstractiMPS
)
    Γ, λ, n = get_data(mps)
    Γ_new = Array{T}.(Γ)
    typeof(mps)(Γ_new, λ, n)
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
    mps[i] = mps.Γ[i], mps.λ[i]/ni
    ni
end

#-------------------------------------------------------------------------------------------
function ent_S(mps::abstractiMPS, i::Integer)
    j = mod(i-1, mps.n) + 1
    ρ = mps.λ[j] .^ 2
    entanglement_entropy(ρ)
end
