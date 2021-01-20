# iTEBD.jl

Julia package for infinite time-evolution block-decimation (iTEBD) calculation.

## Introduction

This julia package is for iTEBD algorithms, introduced by G. Vidal (PRL.91.147902, PRL.98.070201, PRB.78.155117), to simulate time evolution of 1D infinite size systems. The iTEBD algorithm relies on a Trotter-Suzuki and subsequent approximation of the time-evolution operator. It provides an extremely efficient method to study both the short time evolution and the ground state (using imaginary time evolution) of 1-D gapped system.

## Installation

The package can be installed in julia ```REPL```:

```julia
pkg> add https://github.com/jayren3996/iTEBD.jl
```

## Code Examples

### iMPS objects

The states in iTEBD algorithm are represented by infinite matrix-product-states (iMPS):

```julia
struct iMPS{TΓ<:Number, Tλ<:Number}
    Γ::Vector{Array{TΓ, 3}}
    λ::Vector{Vector{Tλ}}
    n::Integer
end
```

Here we use a slightly different representation with that of Vidal's. The tensor ```Γ[i]``` already contain the Schmidt spectrum ```λ[i]```, which means when brought to canonical form, each ```Γ[i]``` is right-canonical, while ```λ[i]``` contains the entanglement information.

There is a function ```rand_iMPS(n::Integer,d::Integer,dim::Integer)``` that generates a random iMPS with ```n``` periodic sites, ```d``` local degrees of freedom, and bond dimension ```dim```.

With a set of given tensors, a ```iMPS``` object can be explicitly constructed by ```iMPS(Γ)```.

### Hamiltonian

An Hamiltonian is just an  ```Array{T,2}```. There is also a helper function ```spinmat``` for constructing spin Hamiltonian. For example, AKLT Hamiltoniancan be constructed by:

```julia
hamiltonian = begin
    SS = spinmat("xx", 3) + spinmat("yy", 3) + spinmat("zz", 3)
    SS + 1/3 * SS^2 - 2/3 * spinmat("11", 3)
end
```

### Setup and run iTEBD

The iTEBD algorithm is generated by an ```iTEBD_Engine``` object:

```julia
struct iTEBD_Engine{T<:AbstractMatrix}
    gate ::T
    renormalize::Bool
    bound::Int64
    tol  ::Float64
end
```

where ```gate``` is the local time-evolving operator, ```renormalize``` controls whether to renorm the Schmidt spectrum in the simulation (which is necessary in the non-unitary evolution, such as imaginary-time iTEBD), ```bound``` controls the truncation of the singular values, and ```tol``` controls the minimal value of the singular value below witch the singular value will be discarded.

The ```iTEBD_Eigine``` can be constructed by the function ```itebd```:

```julia
function itebd(
    H::AbstractMatrix{<:Number},
    dt::AbstractFloat;
    mode::String="r",
    renormalize::Bool=true,
    bound::Int64=BOUND,
    tol::Float64=SVDTOL
)
```

We show an explict example to solve the ground state of AKLT model:

```julia
using iTEBD

# Create random iMPS
imps = begin
    dim_num = 50
    rand_iMPS(2, 3, dim_num)
end

# Create AKLT Hamiltonian and iTEBD engine
hamiltonian = begin
    SS = spinmat("xx", 3) + spinmat("yy", 3) + spinmat("zz", 3)
    SS + 1/3 * SS^2 + 2/3 * spinmat("11", 3)
end

engine = begin
    time_step = 0.01
    itebd(hamiltonian, time_step, mode="i")
end

# Exact AKLT ground state
aklt = begin
    aklt_tensor = zeros(2,3,2)
    aklt_tensor[1,1,2] = +sqrt(2/3)
    aklt_tensor[1,2,1] = -sqrt(1/3)
    aklt_tensor[2,2,2] = +sqrt(1/3)
    aklt_tensor[2,3,1] = -sqrt(2/3)
    aklt_tensor
    iMPS([aklt_tensor, aklt_tensor])
end

# Setup TEBD
for i=1:2000
    global imps, aklt, engine
    imps = engine(imps)
    if mod(i, 100) == 0
        println("Overlap: ", inner_product(aklt, imps))
    end
end
```

Here we calculate the inner product of intermediate state and the exact AKLT ground state. We see the overlap quickly converges to 1.

### Canonical form

In many cases, it is much simpler to work on the canonical form of MPS. Here, the canonical form is the right-canonical form. However, we keep track of the Schmidt values so that it can easily transformed to Schmidt canonical form.

The canonical form is obtained using the function ```canonical(imps::iMPS)```. Note that this function only works when the transfer matrix has single fixed points. Otherwise we should first block diagonalized the tensor using the function ```block_canonical(Γ::AbstractArray{<:Number, 3})```.
