module Chebyshev

using LinearAlgebra, FFTW, ToeplitzMatrices

@inline ChebΘ(N) = (0:N) .* (π / N)
"""
    ChebPoint(N)

Return Chebyshev point of `N+1`.
"""
@inline ChebPoint(N) = cos.(ChebΘ(N))

export ChebPoint

"""
    ChebDiffMat(N)

Compute Chebyshev differention matrix and Chebyshev collocation point of the size `N+1`.
"""
function ChebDiffMat(N)
    if N==0 
        return zeros(1,1), [1]
    end

    c(ii) = ifelse(0<ii<N, 1, 2)

    D = zeros(N+1, N+1)
    x = ChebPoint(N)

    # Off-diagonal
    for i∈0:N
        for j∈i+1:N
            I, J = i+1, j+1
            # upper triangle
            D[I,J] = c(i)/c(j)*((-1)^(i+j))/(x[I]-x[J])
            # lower triangle
            D[N+2-I,N+2-J] = -D[I,J]
        end
    end
    
    # diagonal
    D[1,1] = (2N^2+1)/6
    D[N+1,N+1] = -D[1,1]
    for i∈1:N-1 
        I=i+1
        D[I,I] = -x[I]/(2*(1-x[I]^2)) 
    end

    return D, x
end

export ChebDiffMat

"""
    SincDiff1(N)

Compute first Fourier differention matrix with periodic Sinc method.
"""
function SincDiff1(N)
    Δx = 2π/N
	vr = [ifelse(i==1, 0., (-1)^i * cot((i-1)*Δx/2)/2) for i∈1:N]
	vc = [ifelse(i==1, 0.,-(-1)^i * cot((i-1)*Δx/2)/2) for i∈1:N]
	return Toeplitz(vc,vr)
end

"""
    SincDiff2(N)

Compute second Fourier differention matrix with periodic Sinc method.
"""
function SincDiff2(N)
    Δx = 2π/N
	vr = [ifelse(i==1, -π^2/3Δx^2 - 1/6, (-1)^i/sin((i-1)*Δx/2)^2/2) for i∈1:N]
	return Toeplitz(vr,vr)
end

export SincDiff1, SincDiff2


"""
    polyInterp(t, y)

Construct a callable polynomial interpolant through the points in
vectors `t`,`y` using the barycentric interpolation formula.
Inspired from [this website](https://tobydriscoll.net/fnc-julia/globalapprox/barycentric.html#function-polyinterp)
"""
function polyInterp(t, y)
    N = length(t) - 1
    C = (t[end] - t[1])/8  # scale
    tc = t/C

    ω = ones(N+1)
    for i∈2:N+1
        d = tc[1:i-1] .- tc[i]
        @. ω[1:i-1] *= d
        ω[i] = prod(-d)
    end
    a = 1 ./ω

    function p(x)
        terms = @. a/(x - t)
        if any(isinf.(terms))
            idx = findfirst(x.==t)
            return y[idx]
        end
        return sum(y .* terms) / sum(terms)
    end

    return p
end

export polyInterp


"""
    ChebDiffFFT(v)

Evaluate Chebyshev differentiation of `v` using FFT and iFFT.
"""
function ChebDiffFFT(v)
    N = length(v) - 1
    x = ChebPoint(N)

    V = Vector{eltype(v)}(undef, 2N)
    w = Vector{eltype(v)}(undef,  N+1)
    ii = 0:N-1

    V[1:N+1] .= v[1:N+1]
    @inbounds for i in 1:N-1
        V[N+i+1] = v[N-i+1]  # manually reversing v[2:N]
    end

    # real fft (cosine)
    k = fftfreq(2N,2N)
    U = real.(fft(V))
    W = real.(ifft(1im * k .* U)) 

    @. w[2:N] = -W[2:N] / sqrt(1-x[2:N]^2)
    w[1] = sum(@. ii^2 * U[ii+1])/N + 0.5*N*U[N+1]
    w[N+1] = sum(@. (-1)^(ii+1) * ii^2 * U[ii+1])/N + 0.5*(-1)^(N+1)*N*U[N+1]

    return w
end

export ChebDiffFFT

"""
    Cheb2Poly(N)

    Given ∑ₙ aₙ Tₙ(x) = ∑ₙ cₙ xⁿ, where n=0...N, try to find a matrix A such that c = Aa.
"""
function Cheb2PolyMat(N)
    Ng = N+1
    A = zeros(Ng, Ng)
    for i∈1:Ng
        # initial 2 cases for recurence
        (i==1 || i==2) && (A[i,i] = 1; continue)

        # general recurence
        # Tᵢ = 2xTᵢ₋₁ - Tᵢ₋₂
        for j∈1:i
            A[i,j] = (j==1 ? 0 : 2A[i-1,j-1]) - A[i-2,j]
        end
    end
    return A
end
export Cheb2PolyMat

"""
    ChebyODEQuad(N)

    Return weight and points deducted from ODE style of Chebyshev differentiation matrix. 
    This sacrifice one order as the point at x = -1 is sacrificed.
"""
function ChebyODEQuad(N)
    D,x = ChebDiffMat(N)
    xI = x[1:end-1]
    D⁻¹= D[1:end-1,1:end-1]^-1
    w = D⁻¹[1,:]
    return w, xI
end
export ChebyODEQuad

"""
    ClenshawCurtis(N)

    Return weight and points of Clenshaw-Curtis quadrature.
"""
function ClenshawCurtisQuad(N)
    w = zeros(N+1); v = ones(N-1)
    θ = ChebΘ(N)[2:end-1]
    if mod(N,2) == 0
        w[1] = w[end] = 1/(N^2-1)
        for k∈1:(N÷2-1) 
            @. v -= 2cos(2k*θ)/(4k^2-1)
        end
        @. v -= cos(N*θ)/(N^2-1)
    else
        w[1] = w[end] = 1/(N^2)
        for k∈1:(N-1)÷2 
            @. v -= 2cos(2k*θ)/(4k^2-1)
        end
    end
    w[2:end-1] .= 2v/N

    return w, ChebPoint(N)
end
export ClenshawCurtisQuad

"""
    GaussLegendreQuad(N)

    Return weight and points of Gauss-Legendre quadrature.
"""
function GaussLegendreQuad(N)
    β = 0.5 ./ sqrt.(1 .- (2*(1:N-1)).^(-2))
    T = diagm(1 => β) + diagm(-1 => β)
    D, V = eigen(T)
    x = D # root of legendre polynomial
    i = sortperm(x)
    x = x[i]
    w = 2 * V[1, i].^2
    return w, x
end
export GaussLegendreQuad

end