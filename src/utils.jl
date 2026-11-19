@static if VERSION >= v"1.11"
    using Base.ScopedValues
else
    using ScopedValues
end

let DEFAULT_WORKERS = Threads.Atomic{Int}(1), SCOPED_WORKERS = ScopedValue(0)
    global function get_num_threads()
        val = SCOPED_WORKERS[]
        return val == 0 ? DEFAULT_WORKERS[] : val
    end
    global function set_num_threads(n::Int)
        n > 0 || throw(ArgumentError(lazy"Cannot set worker count to $n"))
        DEFAULT_WORKERS[] = n
    end
    set_num_threads(::Nothing) = set_num_threads(max(Threads.nthreads() ÷ 2, 1))
    function set_num_threads(f::Function, n::Int)
        n > 0 || throw(ArgumentError(lazy"Cannot set worker count to $n"))
        with(f, SCOPED_WORKERS => n)
    end
    set_num_threads(f::Function, ::Nothing) =
        set_num_threads(f, max(Threads.nthreads() ÷ 2, 1))
end

_device_eigen(::AbstractArray) = false
device_eigen(adapter) = _device_eigen(adapt(adapter, zeros(Float32, 1, 1)))

diameter(sz::NTuple{2, Int}) = sz[1] == sz[2] ? sz[1] :
    throw(ArgumentError(lazy"Expected a square grid, got $sz"))
diameter(A) = diameter(size(A)[1:2])

const Maybe{T} = Union{T,Nothing}
