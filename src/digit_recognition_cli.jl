push!(LOAD_PATH, "./src")

using ArgParse
using image_transform
using neural_network
using LinearAlgebra

function get_parsed_args()
    settings = ArgParseSettings(
        description="A command line tool to recognise a handwritten digit from an image."
    )
    @add_arg_table settings begin
        "path"
            help = "path to the image"
            required = true
    end
    return parse_args(settings)
end

function main()
    parsed_args = get_parsed_args()
    path = parsed_args["path"]
    image = image_to_vector(path)

    network = NeuralNetwork([784, 30, 16, 10], "./src/params")
    feed_forward!(network, image)
    total = sum(network.layers[end])
    predictions = sort([zip(network.layers[end] ./ total, 0:9)...], by = pair -> pair[1], rev = true)
    for (probability, label) ∈ predictions
        println("$(label): $(round(probability * 100, digits=2))%")
    end
end

main()
