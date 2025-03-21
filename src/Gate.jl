#-------------------------------------------------------------------------------------------
# QUANTUM GATE
#-------------------------------------------------------------------------------------------
"""
    tensor_applygate!(G, Γs, λl; keywords...)

Apply Gate:
    |
    G
    |              |   |        |
  --Γ--  ==> --λl--Γ₁--Γ₂-- ⋯ --Γₙ--,
where:
    |          |
  --Γₙ--  =  --Aₙ--λₙ--

Return list tensor list [Γ₁,⋯,Γₙ], and values list [λ₁,⋯,λₙ₋₁].
"""
function tensor_applygate!(
    G::AbstractMatrix{<:Number},
    Γs::AbstractVector{<:AbstractArray{<:Number}},
    λl::AbstractVector{<:Number};
    maxdim=MAXDIM,
    cutoff=CUTOFF,
    renormalize=false
)
    n = length(Γs)
    Γ = tensor_group(Γs)
    tensor_lmul!(λl, Γ)
    GΓ = tensor_umul(G, Γ)
    tensor_decomp!(GΓ, λl, n; maxdim, cutoff, renormalize)
end

#-------------------------------------------------------------------------------------------
export applygate!
function applygate!(
    ψ::abstractiMPS,
    G::AbstractMatrix,
    i::Integer,
    j::Integer;
    maxdim=MAXDIM,
    cutoff=CUTOFF,
    renormalize=false
)
    inds = j>i ? collect(i:j) : [i:ψ.n; 1:j]
    Γs = ψ.Γ[inds]
    λl = ψ.λ[mod(i-2,ψ.n)+1] # the left bond of Γ used
    Γs, λs = tensor_applygate!(G, Γs, λl; maxdim, cutoff, renormalize)
    push!(λs, ψ.λ[j])
    for i in eachindex(inds)
        ψ[inds[i]] = Γs[i], λs[i]
    end
end
