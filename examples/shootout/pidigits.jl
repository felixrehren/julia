load("timing.jl")
load("bigint.jl")

function pidigits(N::Int, printOut::Bool)
    """
    See http://shootout.alioth.debian.org/u64q/performance.php?test=pidigits#about

    Transliterated from Mario Pernici Python's program


    INPUT:

    - N -- a positive integer giving the number of digits of pi to be computed

    - printOut -- a boolean specifying if we want intermediate printouts of digits in packets of 10

    OUTPUT:

    - returns the last ten digits anyway
    
    - prints all the digits in packets of 10 iff printOut == true

    """

    i = k = ns = 0
    k1 = 1
    n,a,d,t,u = map(BigInt,(1,0,1,0,0))

    while true
        k += 1
        t = lshift(n,uint(1))
        n *= k
        a += t
        k1 += 2
        a *= k1
        d *= k1

        if a >= n
            t,u = divmod(n*3 +a, d)
            u += n
            if d > u
                ns = ns*10 + t
                i += 1
                if mod(i,10) == 0
                    if printOut
                        show(ns)
                        @printf("\t:%d\n", i)
                    end
                    if i >= N
                        return ns
                    end
                    ns = 0
                end
                a -= d*t
                a *= 10
                n *= 10

            end
        end
    end
end

function pidigits(N::Int)
    pidigits(N,false)
end


@assert pidigits(1000) == 9216420198

if length(ARGS)==1
    N = int(ARGS[1])
else
    N = 1000
end
@timeit pidigits(N) "pidigits"


