push!(LOAD_PATH, ".")
using ProgressBars
using NPZ
using Statistics
using neural_network

@doc """
Train the neural network on the entire training set.
"""
function run_training_set()
    @info "start reading training data"
    trainingimages::Matrix{Float64} = npzread("dataset/training_images.npy")
    traininglabels::Vector{Int} = parse.(Int, readlines(open("dataset/training_labels.txt")))
    @info "stop reading"
    epochs = 25
    η = 0.15
    batch_size = 5
    trainer::Trainer = Trainer([784, 16, 16, 10], batch_size)
    losses = Float64[]
    for i ∈ 1:epochs
        @inbounds for j ∈ ProgressBar(1:batch_size:60000)
            image_batch = @view trainingimages[:, j:(j + batch_size-1)]
            labels = traininglabels[j:(j + batch_size-1)]
            l = train!(trainer, image_batch, labels, η)
            push!(losses, l)
        end
        mean_loss = round(mean(@view losses[end-100:end]) * 100, digits=2)
        println("loss after $(i) epoch(s): $(mean_loss)%")
        
    end

    save_params(trainer.network)
end

@time run_training_set()
