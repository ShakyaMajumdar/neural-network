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
σ_prime(y::AbstractFloat) = 0.25 # y * (1 - y)

η = 0.05 # learning rate

function train(image, label)
    # input layer
    layers[1] .= image

    # feed forward
    for i ∈ 2:length(neuron_counts)
        layers[i] .= σ.((weights[i] * layers[i - 1]) .+ biases[i])
    end

    # prediction is output layer neuron with max activation
    prediction = findmax(layers[end])

    # backpropagation
    target = [i == label ? 1. : 0. for i ∈ 0:9]

    @debug "" predict = (prediction[2] - 1) actual=label confidence_actual=layers[end][label + 1] confidence_predict=prediction[1]

    delc_dela = 2 .* (layers[end] .- target)
    for L ∈ length(neuron_counts):-1:2
        delc_delw = map(Iterators.product(1:neuron_counts[L], 1:neuron_counts[L-1])) do (j, k)
            delc_dela[j] * σ_prime(layers[L][j]) * layers[L-1][k]
        end

        delc_delb = delc_dela .* σ_prime.(layers[L])

        delc_dela = [
            sum(
                [
                    weights[L][j, k] * σ_prime(layers[L][j]) * delc_dela[j]
                    for j ∈ 1:neuron_counts[L]
                ]
            )
            for k ∈ 1:neuron_counts[L-1]
        ]
        # update weights and biases at this layer
        biases[L] .+= -η * delc_delb
        weights[L] .+= -η * delc_delw
    end
end


@info "start reading training data"
trainingimages = npzread("dataset/training_images.npy")
traininglabels = parse.(Int, readlines(open("dataset/training_labels.txt")))
@info "stop reading"


for i ∈ 1:60000
    train(trainingimages[i, :], traininglabels[i])
    if i % 600 == 0
        @info "$(i / 600)%"
    end
end

mkpath("src/params")
@info "saving weights"
for (i, weight) in enumerate(weights)
    npzwrite("src/params/weight$(i).npy", weight)
end
@info "saved weights"


@info "saving biases"
for (i, bias) in enumerate(biases)
    npzwrite("src/params/bias$(i).npy", bias)
end
@info "saved biases"

end
