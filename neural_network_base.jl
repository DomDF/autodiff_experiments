using LinearAlgebra, ForwardDiff    # for backpropagation
using Random                        # for randomly initialising the weights and biases

# Define the sigmoid activation function
function sigmoid_activation(x)
    return 1 / (1 + ℯ^(-x))
end

function relu_activation(x)
    return max(0, x)
end

# Define the structure of our neural network
mutable struct neural_network
    input_dim::Int
    hidden_dim::Int
    output_dim::Int
    W₁::Array{Float64,2}
    b₁::Array{Float64,1}
    W₂::Array{Float64,2}
    b₂::Array{Float64,1}
end

function initialise_network(input_size::Int, hidden_size::Int, output_size::Int; prng = MersenneTwister(240819))
    return neural_network(input_size, hidden_size, output_size, 
                          randn(prng, (input_size, hidden_size)), zeros(hidden_size),
                          randn(prng, (hidden_size, output_size)), zeros(output_size))
end

# Define the forward propagation process
function forward_prop(nn::neural_network, a₁::Array{Float64,2})
    a₂ = b₁ .+ a₁ .* nn.W₁ |> 
        z -> sigmoid_activation.(z)
    ŷ = b₂ .+ a₂ .* nn.W₂ |>
        z -> sigmoid_activation.(z)
    return a₁, a₂, ŷ
end

# Define the loss function
function mse_loss(nn::neural_network, a₁::Array{Float64,2}, y::Array{Float64,2})
    ŷ, _ = forward(nn, X)
    return sum((y .- yHat).^2) / length(y)
end

# Define the training process
function train(nn::SimpleNN, X::Array{Float64,2}, y::Array{Float64,2}, epochs::Int, learning_rate::Float64)
    for i in 1:epochs
        # Compute the gradient of the loss function with respect to the weights
        grad = ForwardDiff.gradient(w -> loss(SimpleNN(nn.inputLayerSize, nn.hiddenLayerSize, nn.outputLayerSize, w[1:size(nn.W1)...], w[(size(nn.W1) + 1):end]...), X, y), vcat(nn.W1[:], nn.W2[:]))

        # Update the weights using the computed gradient
        nn.W1 -= reshape(grad[1:length(nn.W1)], size(nn.W1)) * learning_rate
        nn.W2 -= reshape(grad[(length(nn.W1) + 1):end], size(nn.W2)) * learning_rate
    end
end

# Now let's initialize our NN and train it
X = randn(100, 3) # Inputs
y = randn(100, 1) # Targets
nn = SimpleNN(3, 5, 1) # Initialize our network

# Training process
train(nn, X, y, 1000, 0.01)
