#---------------------------------------------------------------------------------------------------
# Schmidt Canonical Form
#---------------------------------------------------------------------------------------------------
"""
schmidt_canonical(Γ; kerwords)

Schmidt Canonical Form
1. Given a right canonical form, return a Schmidt canonical form.
2. This algorithm assume there is no degeneracy.
"""
function schmidt_canonical(
    Γ::AbstractArray{<:Number,3};
    krylov_power::Integer=KRLOV_POWER,
    maxdim=MAXDIM,
    cutoff=SVDTOL,
    renormalize=false
)
    X, Yt = begin
        R = steady_mat(Γ, krylov_power=krylov_power)
        L = steady_mat(Γ, krylov_power=krylov_power, dir=:l)
        R_res = cholesky(R)
        L_res = cholesky(L)
        R_res.L, L_res.U
    end
    U, S, V = svd_trim(Yt * X; maxdim, cutoff, renormalize)
    R_mat = inv(Yt) * U * Diagonal(S)
    L_mat = V * inv(X)
    Γ_new = canonical_gauging(Γ, R_mat, L_mat)
    Γ_new, S
end
#---------------------------------------------------------------------------------------------------
# Multiple tensors
function schmidt_canonical(
    Γs::AbstractVector{<:AbstractArray{<:Number, 3}};
    krylov_power::Integer=KRLOV_POWER,
    renormalize::Bool=false,
    bound::Integer=BOUND,
    tol::Real=SVDTOL,
    zerotol::Real=ZEROTOL
)
    n = length(Γs)
    Γ_grouped = tensor_group(Γs)

    A, λ = schmidt_canonical(
        Γ_grouped, 
        krylov_power=krylov_power, 
        renormalize=renormalize,
        bound=bound,
        tol=tol,
        zerotol=zerotol
    )
    tensor_lmul!(λ, A)
    tensor_decomp!(A, λ, λ, n; maxdim, cutoff, renormalize)
end
#---------------------------------------------------------------------------------------------------
export canonical_trim
function canonical_trim(
    Ts::AbstractVector{<:AbstractArray{<:Number, 3}};
    renormalize=true,
    bound::Integer=BOUND,
    tol::Real=SVDTOL
)
    n = length(Ts)
    T = tensor_group(Ts)
    T_RC = right_canonical(T)
    T_BRC = block_trim(T_RC)
    A, λ = schmidt_canonical(T_BRC, renormalize=renormalize)
    tensor_lmul!(λ, A)
    tensor_decomp!(A, λ, λ, n; maxdim, cutoff, renormalize)
end
