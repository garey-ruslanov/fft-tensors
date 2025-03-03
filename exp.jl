
function read_tensor(filename::String)
    input = split(open((io)->read(io, String), filename))
    d = parse(Int, input[1])
    shape = Tuple(map((x; t=Int)->parse(t, x), input[2:d+1]))
    arr = reshape(map((x; t=Float64)->parse(t, x), input[d+2:end]), shape)
    return arr
end

tensor = read_tensor("tensor.txt")
println(tensor, '\n')
println(size(tensor), '\n')
println(ndims(tensor))
