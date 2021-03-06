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
    labels = 0:9
    network = NeuralNetwork([784, 16, 16, 10], "./src/params")
    feed_forward!(network, image)
    probabilities = normalize(network.layers[end], 1)
    predictions = sort(zip(probabilities, labels) |> collect, by=first, rev=true)
    for (probability, label) ∈ predictions
        println("$(label): $(round(probability * 100, digits=2))%")
    end
end

main()
