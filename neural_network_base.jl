using LinearAlgebra, ForwardDiff, ReverseDiff, Zygote   # for backpropagation
using Random                                            # for randomly initialising the weights and biases

# Define the sigmoid activation function
function sigmoid_activation(x)
    return 1 / (1 + ℯ^(-x))
end

function relu_activation(x)
    return max(zero(T), x)
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

function initialise_network(input_size::Int, hidden_size::Int, output_size::Int; prng = MersenneTwister(240819))
    return neural_network(input_size, hidden_size, output_size, 
                          randn(prng, (input_size, hidden_size)), zeros(hidden_size),
                          randn(prng, (hidden_size, output_size)), zeros(output_size))
end

# Define forward propagation process
function forward_prop(nn::neural_network, a₁::Array{Float64,2})
    a₂ = [reshape(a₁[i, :], (1, nn.input_dim)) * nn.W₁ .+ reshape(nn.b₁, (1, nn.hidden_dim)) for i ∈ 1:size(a₁)[1]] |> 
        z -> [sigmoid_activation.(z[i]) for i ∈ 1:length(z)]
    ŷ = [a₂[i] * nn.W₂ .+ reshape(nn.b₂, (1, nn.output_dim)) for i ∈ 1:size(a₂)[1]] |>
        z -> [sigmoid_activation.(z[i])[1] for i ∈ 1:length(z)]
    return a₁, a₂, ŷ
end

# Using a mse loss function
function mse_loss(nn::neural_network, a₁::Array{Float64, 2}, y::Array{Float64})
    _, _, ŷ = forward_prop(nn, a₁)
    return sum((y .- ŷ).^2) / length(y)
end

# Define the training (backpropagation) process
function train(nn::neural_network, a₁::Array{Float64,2}, y::Array{Float64,2}; n_epochs::Int = 10, η::Float64 = 0.01)
    for i in 1:n_epochs
        # Compute the gradient of the loss function with respect to the weights and biases
        ∇p = ForwardDiff.gradient(mse_loss(nn, a₁, y), [nn.W₁; nn.b₁; nn.W₂; nn.b₂])

        # Update the weights using the computed gradient
        nn.W₁ -= reshape(∇p[begin:length(nn.W₁)], size(nn.W₁)) * η
        nn.b₁ -= reshape(∇p[(length(nn.W₁) + 1):(length(nn.W₁) + length(nn.b₁))], size(nn.b₁)) * η
        nn.W₂ -= reshape(∇p[(length(nn.W₁) + length(nn.b₁) + 1):(length(nn.W₁) + length(nn.b₁) + length(nn.W₂))], size(nn.W₂)) * η
        nn.b₂ -= reshape(∇p[end-length(nn.b₂):end], size(nn.b1)) * η
    end
end

mlp = initialise_network(2, 5, 1) # Initialize network

nn = mlp

a₁ = MersenneTwister(240819) |> prng -> randn(prng, (100, 2))   # Inputs
y = 2 .+ 3 .* a₁[:,1] .+ 5 .* a₁[:,2]                           # Targets

# Training process
train(nn, X, y, 1000, 0.01)
