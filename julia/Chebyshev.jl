module Chebyshev

using LinearAlgebra

"""
    ChebDiffMat(N)

Compute Chebyshev differention matrix and Chebyshev collocation point of the size `N+1`
"""
function ChebDiffMat(N)
    if N==0 
        return zeros(1,1), [1]
    end

    c(ii) = ifelse(0<ii<N, 1, 2)

    D = zeros(N+1, N+1)
    x = cos.((N:-1:0) .* (π / N))

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
    D[1,1] = -(2N^2+1)/6
    D[N+1,N+1] = -D[1,1]
    for i∈1:N-1 
        I=i+1
        D[I,I] = -x[I]/(2*(1-x[I]^2)) 
    end

    return D, x
end

export ChebDiffMat

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

end