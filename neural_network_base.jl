using ForwardDiff, ReverseDiff, Zygote   # for backpropagation
using LinearAlgebra, Random              # for initialising parameters
using Statistics, DataFrames             # for storing data

# Some activation functions
function sigmoid_activation(x)
    return 1 / (1 + exp(-x))
end

function relu_activation(x)
    return max(zero(T), x)
end

function linear_activation(x)
    return x
end

# Define the structure of our neural network
mutable struct neural_network
    input_dim::Int
    hidden_dim::Int
    output_dim::Int
    W₁::Array{Float64, 2}
    b₁::Array{Float64, 1}
    W₂::Array{Float64, 2}
    b₂::Array{Float64, 1}
end

# Set up a network with inital weights sampled from a std normal distribution and initial biases set to zero
function initialise_network(input_size::Int, hidden_size::Int, output_size::Int; prng = MersenneTwister(240819))
    return neural_network(input_size, hidden_size, output_size, 
                          randn(prng, (input_size, hidden_size)), zeros(hidden_size),
                          randn(prng, (hidden_size, output_size)), zeros(output_size))
end

# Using a mean squared error loss function
function mse(ŷ::Array{Float64}, y::Array{Float64})
    return (y .- ŷ).^2 |> se -> mean(se)
end

# Make predictions using the current parameters and compute the loss
function find_loss(nn::neural_network, a₁::Array{Float64}, y::Array{Float64})
    a₂ = [LinearAlgebra.dot(a₁[j, :], nn.W₁[:, i]) .+ nn.b₁[i] for j ∈ 1:size(a₁)[1], i ∈ 1:nn.hidden_dim] |>
        z -> [sigmoid_activation.(z[i, :]) for i ∈ 1:size(z)[1]]
    
    ŷ = [LinearAlgebra.dot(a₂[j], nn.W₂) .+ nn.b₂ for j ∈ 1:size(a₂)[1]] |>
        z -> [linear_activation(z[i][1]) for i ∈ 1:length(z)]
    
    return mse(ŷ, y), ŷ, a₂
end

# Define the training (backpropagation) process
function train(nn::neural_network, a₁::Array{Float64,2}, y::Array{Float64}; n_epochs::Int = 10, η::Float64 = 0.1)
    
    training_df = DataFrame(epoch = Int[], loss = Float64[])

    for i in 1:n_epochs
        # Compute the gradient of the loss function with respect to the paramneters of the network
        ∇p = Zygote.gradient(p -> find_loss(p, a₁, y)[1], nn)[1]

        # Update the weights using the computed gradient, ∇p (direction) and the learning rate, η (magnitude)
        nn.W₁ -= ∇p.W₁ .* η; nn.b₁ -= ∇p.b₁ .* η
        nn.W₂ -= ∇p.W₂ .* η; nn.b₂ -= ∇p.b₂ .* η

        # Store the loss for each epoch
        append!(training_df, DataFrame(epoch = i, loss = find_loss(nn, a₁, y)[1]))
    end

    return (nn, training_df)
    
end

# Define the function that we want the network to approximate
function gen_data(a₁::Array{Float64, 2})
    return 2 .+ 3 .* a₁[:,1] .+ 5 .* a₁[:,2]
end

# Initialize network and generate inputs (a₁) and targets (y)
mlp = initialise_network(2, 8, 1) 
a₁ = MersenneTwister(240819) |> prng -> randn(prng, (100, mlp.input_dim)) ; y = gen_data(a₁)    # Targets

# Return the trained network parameters and the loss for each epoch
trained_net, training_df = train(mlp, a₁, y, n_epochs = 500, η = 0.0001)

using Plots
Plots.plot(training_df.epoch, training_df.loss, xlabel = "Epoch", ylabel = "Loss", label = "Training Loss")