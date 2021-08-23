module neural_network

using NPZ

neuron_counts = [784, 30, 16, 10]

layers = [Vector{Float64}(undef, neuron_count) for neuron_count ∈ neuron_counts]

weights = [0., [
    (rand(Float64, (next_layer_neuron_count, prev_layer_neuron_count)) .* 10) .- 5
    for (next_layer_neuron_count, prev_layer_neuron_count) ∈ zip(neuron_counts[2:end], neuron_counts)
]...]
biases = [0., [rand(neuron_count) .* 20 .- 10 for neuron_count ∈ neuron_counts[2:end]]...]

σ(x::AbstractFloat) = 1 / (1 + exp(-x))

function train(image, label)
    # input layer
    layers[1] .= image

    # feed forward
    for i ∈ 2:length(neuron_counts)
        layers[i] .= σ.((weights[i] * layers[i - 1]) .+ biases[i])
    end

    # prediction is output layer neuron with max activation
    prediction = findmax(layers[end])
end


println("start reading training data")
trainingimages = npzread("dataset/training_images.npy")
traininglabels = parse.(Int, readlines(open("dataset/training_labels.txt")))
println("stop reading")


for i ∈ 1:60000
    train(trainingimages[i, :], traininglabels[i])
    if i % 600 == 0
        println("$(i / 600)%")
    end
end

end
