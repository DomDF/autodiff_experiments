################################################################
#
# Loading libraries
#
################################################################

using Zygote                            # for differentiating loss function
using LinearAlgebra, Random             # for initialising parameters
using Statistics, DataFrames            # for storing data

################################################################
#
# Defining activation functions
#
################################################################

function sigmoid_activation(x)
    return 1 / (1 + exp(-x))
end

function relu_activation(x)
    # Compare the zero of a matching type
    T = typeof(x)  
    return max(zero(T), x)
end

function linear_activation(x)
    return x
end

function softmax_activation(x::Vector)
    return exp.(x) ./ sum(exp.(x))
end

activations_dict = Dict(
    "sigmoid" => sigmoid_activation,
    "relu" => relu_activation,
    "linear" => linear_activation,
    "softmax" => softmax_activation
)

################################################################
#
# Defining loss functions
#
################################################################

function mse_loss(ŷ::Array{Float64}, y::Array{Float64})
    return (y .- ŷ).^2 |> mean
end

function cross_entropy_loss(y_true::Vector, y_pred::Vector)
    # Ensure numerical stability by avoiding taking the log of 0 or 1
    ϵ = 10^-12
    clipped_preds = clamp.(y_pred, ϵ, 1 - ϵ)
    
    # ℒ(y, ŷ) = -Σ[yᵢ × log(ŷᵢ)]
    return -sum(y_true .* log.(clipped_preds))
end

loss_dict = Dict(
    "mse" => mse_loss,
    "cross_entropy" => cross_entropy_loss
)

################################################################
#
# Functions to initialise a neural network
#
################################################################

mutable struct neural_network
    input_dim::Int
    hidden_dims::Vector{Int}
    output_dim::Int
    Ws::Vector{Array{Float64, 2}}
    bs::Vector{Array{Float64, 1}}
end

function initialise_network(input_size::Int, hidden_sizes::Vector{Int}, output_size::Int; prng=MersenneTwister(240819))
    Ws = []; bs = []

    network_dims = [input_size; hidden_sizes; output_size]

    for i = 1:length(network_dims)-1
        push!(Ws, randn(prng, network_dims[i], network_dims[i+1]))
        push!(bs, zeros(network_dims[i+1]))
    end

    return neural_network(input_size, hidden_sizes, output_size, Ws, bs)
end

################################################################
#
# Forward propagation and loss calculation
#
################################################################

function forward_prop(nn::neural_network, a, layer_idx = 1; 
                      hidden_activation::String = "sigmoid", output_activation::String = "linear")
    @assert layer_idx > 0 "layer_idx must be greater than 0"
    @assert hidden_activation ∈ keys(activations_dict) "hidden_activation must be one of $(keys(activations_dict))"
    @assert output_activation ∈ keys(activations_dict) "output_activation must be one of $(keys(activations_dict))"

    # check for the last layer
    if layer_idx > length(nn.Ws)
        return [a]
    end

    act_h = activations_dict[hidden_activation]; act_o = activations_dict[output_activation]

    if layer_idx == length(nn.Ws)
        # use a linear activation function for the last layer
        next_a = act_o.(a * nn.Ws[layer_idx] .+ nn.bs[layer_idx]')
    else
        # otherwise use a sigmoid activation function
        next_a = act_h.(a * nn.Ws[layer_idx] .+ nn.bs[layer_idx]')
    end

    # recursively find all the outputs and concatenate them
    return vcat([a], forward_prop(nn, next_a, layer_idx + 1))
end

function find_loss(nn::neural_network, a::Array{Float64}, y::Array{Float64};
                   loss::String = "mse")
    @assert loss ∈ keys(loss_dict) "loss must be one of $(keys(loss_dict))"

    loss_fn = loss_dict[loss]

    activations = forward_prop(nn, a); ŷ = activations[end]

    return loss_fn(ŷ, y), ŷ, activations
end

################################################################
#
# Training (back propagation and updating parameters)
#
################################################################

function train(nn::neural_network, a::Array{Float64}, y::Array{Float64};
               a_test::Array{Float64} = a, y_test::Array{Float64} = y, 
               loss::String = "mse",
               n_epochs::Int = 10, ηᵢ::Float64 = 0.1, scheduler::Bool = false)
    @assert n_epochs > 0 "n_epochs must be greater than 0"

    training_df = DataFrame(epoch = Int[], loss = Float64[], test_loss = Float64[])

    for i in 1:n_epochs

        if scheduler
            η = ηᵢ * (0.5 ^ (i ÷ (n_epochs / 5)))
        else
            η = ηᵢ
        end
        
        ∇p = Zygote.gradient(p -> find_loss(p, a, y, loss = loss)[1], nn)[1]
        
        for j = 1:length(nn.Ws)
            nn.Ws[j] -= η * ∇p.Ws[j]
            nn.bs[j] -= η * ∇p.bs[j]
        end
        
        append!(training_df, 
                DataFrame(epoch = i, 
                          loss = find_loss(nn, a, y)[1],
                          test_loss = find_loss(nn, a_test, y_test)[1]))
    end
    
    return nn, training_df
end

################################################################
#
# For preparing data
#
################################################################

function one_hot_encode(y::Vector)
    # Find the unique values in y
    unique_values = unique(y) |> sort
    
    # Create a dictionary mapping each unique value to an integer
    unique_dict = Dict(unique_values[i] => i for i in 1:length(unique_values))
    
    # Create an array of zeros with the same length as y
    encoded_y = zeros(length(y), length(unique_values))
    
    # For each row in encoded_y, set the value at the index given by unique_dict[y[i]] to 1
    for i in 1:length(y)
        encoded_y[i, unique_dict[y[i]]] = 1
    end
    
    return encoded_y
end
