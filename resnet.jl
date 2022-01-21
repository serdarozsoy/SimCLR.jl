using Knet

"""
Sequential provides chain of its arguments.

Arguments:
- `layers`: layers will be chained

Returns:
- Chain of layers
"""
struct Sequential
    layers
end
Sequential(layers...) = Sequential(layers)
(n::Sequential)(x) = (for l in n.layers; x = l(x); end; x)


"""
Linear layer.

Arguments:
- `in_size::Integer`: input size
- `out_size::Integer`: output size
- `f:Function`: activation function (default:identity)
    
Returns:
- Linear layer function which takes input x
"""
struct Linear
    w
    b
    f
end
function Linear(in_size::Int, out_size::Int; f::Function=identity)
    w = param(out_size, in_size, init=xavier)
    b = param(out_size, 1, init=zeros)
    Linear(w, b, f)
end
function (l::Linear)(x) 
    out = l.f.(l.w * mat(x) .+ l.b)
    return out
end


"""
Convolutional layer.

Arguments:
- `size::Int`: filter (kernel) size
- `in_ch::Int`: input channel size 
- `out_ch::Int`: output channel size
- `pad::Int`: padding size 
- `stride::Int`: stride size 
- `f::Function`: Activation function after convolutional operation (default=identity)
- `bias::Bool`: for whether bias will be used or not (default=true)
    
Returns:
- Convolutional layer function which takes input x
"""
struct Conv2d
    w
    b
    pad
    stride
    f
end
function Conv2d(size::Int, in_ch::Int, out_ch::Int; pad::Int, stride::Int,
    f::Function=identity, bias::Bool=true)
    w = param(size, size, in_ch, out_ch, init=xavier)
    b = nothing
    if bias
        b = param(1, 1, out_ch, 1, init=zeros)
    end
    Conv2d(w, b, pad, stride, f)
end
function (c::Conv2d)(x) 
    out = conv4(c.w, x, padding=c.pad, stride=c.stride)
    if !isnothing(c.b)
        out = out .+ c.b
    end
    out = c.f.(out)
end


"""
Pooling layer.

Arguments:
- `window::Int`: window size for pooling
- `pad::Int`: padding size
- `stride::Int`: stride size
- `mode::Int`: Pooling mode. default=`0`.  
    -   `0` for max pooling, 
    -   `1` for average pooling including padded values, 
    -   `2` for average poolingexcluding padded values, 
    -   `3` for deterministic max pooling.
Returns:
- Pooling layer function which takes input x
"""
struct Pool2d
    window
    pad
    stride
    mode
end
Pool2d(window::Int; pad::Int, stride::Int, mode::Int=0) = Pool2d(window, pad, stride, mode)
(p::Pool2d)(x) = pool(x, window=p.window, padding=p.pad, stride=p.stride,  mode=p.mode)



"""
Global Average Pooling layer.

According to size of input, it determines the window size. 
- `mode::Int`: Pooling mode. default=`1`.  
    -   `1` for average pooling including padded values, 
    -   `2` for average poolingexcluding padded values, 
Returns:
- Global Average Pooling layer function which takes input x
"""
function GlobAvgPool2d(y; mode=1)
    w1 = size(y,1)
    w2 = size(y,2)
    out = pool(y, window=(w1, w2), mode=mode)
end



"""
Transform the input tensor with size([1,1,a,b]) to matrix format with size([a,b])
"""
function flatten(x)
    x = mat(x)
end

"""
Batch Normalization

Arguments:
- `channels::Int`: input channel size for batch normalization
- `f::Function`: activation function after batch normalization (default: `identity`)

Returns:
- Batch normalized x with size (H,W,C,N) for input x with size (H,W,C,N)
"""
struct BatchNorm2d
    bmoments
    bparams
    f
end
function BatchNorm2d(channels::Int, f::Function=identity) 
    bmoments= bnmoments()
    bparams = Knet.atype(bnparams(channels))
    BatchNorm2d(bmoments, bparams, f)
end
function (bn::BatchNorm2d)(x)
    out = batchnorm(x, bn.bmoments, bn.bparams)
    bn.f.(out)
end


"""
Residual Block

Arguments:
- `in_ch::Int`: input channel size 
- `out_ch::Int`: output channel size
- `stride::Int`: Stride size

Returns:
- ReLU Output of sequential module with residual connection
"""
struct ResidualBlock
    sequential_module
    shortcut
end
function ResidualBlock(in_ch::Int, out_ch::Int; stride::Int)
    sequential_module = Sequential([
    Conv2d(3, in_ch, out_ch, pad=1, stride=stride, bias=false),
    BatchNorm2d(out_ch, relu),
    Conv2d(3, out_ch, out_ch, pad=1, stride=1, bias=false),
    BatchNorm2d(out_ch)
    ])
    shortcut = identity
    if stride != 1 || in_ch != out_ch
        shortcut = Sequential([
                Conv2d(1, in_ch, out_ch, pad=0, stride=stride, bias=false),
                BatchNorm2d(out_ch)
                ])
    end
    ResidualBlock(sequential_module, shortcut)
end
function (rb::ResidualBlock)(x)
    residual = rb.shortcut(x) 
    out = rb.sequential_module(x)
    out = relu.(out + residual)
end


"""
Bottleneck Block

Arguments:
- `in_ch::Int`: input channel size 
- `out_ch::Int`: output channel size
- `stride::Int`: Stride size

Returns:
- ReLU Output of sequential module with residual connection

"""
struct BottleneckBlock
    sequential_module
    shortcut
end
function BottleneckBlock(in_ch::Int, out_ch::Int; stride::Int)
    expansion = 4
    sequential_module = Sequential([
    Conv2d(1, in_ch, out_ch, pad=0, stride=1, bias=false),
    BatchNorm2d(out_ch, relu),
    Conv2d(3, out_ch, out_ch, pad=1, stride=stride, bias=false),
    BatchNorm2d(out_ch, relu),
    Conv2d(1, out_ch, expansion*out_ch, pad=0, stride=1, bias=false),
    BatchNorm2d(expansion*out_ch)
    ])

    shortcut = identity
    if stride != 1 || in_ch != expansion*out_ch
        shortcut = Sequential([
                Conv2d(1, in_ch, expansion*out_ch, pad=0, stride=stride, bias=false),
                BatchNorm2d(expansion*out_ch)
                ])
    end

    BottleneckBlock(sequential_module, shortcut)
end
function (bb::BottleneckBlock)(x)
    residual = bb.shortcut(x) 
    out = bb.sequential_module(x)
    out = relu.(out + residual)
end



"""
Block Group

If `Bottleneck Block` is selected as block function, `expansion` factor is 4, otherwise expansion factor is 1.

Stride of first layer is pre-selected `stride` from architecture, stride of other layers is 1.

Arguments:
- `in_ch::Int`: input channel size 
- `out_ch::Int`: output channel size
- `blocks::Int`: Number of block layers will be generated for this group
- `stride::Int`: Stride size

Returns:
- Sequential group of layers
"""
function block_group(block_fn, in_ch::Int, out_ch::Int, blocks::Int; stride::Int)
    block_fn==BottleneckBlock ? expansion=4 : expansion=1
    layers = []
    push!(layers, block_fn(in_ch, out_ch, stride=stride))
    for i in 2:blocks
        push!(layers, block_fn(expansion*out_ch, out_ch, stride=1))
    end
    return Sequential(layers)
end


"""
Create ResNet model according to selected Resnet architecture.

Arguments:
- `layers::Array{Int, 1}`: Number of layers in each block groups according to selected ResNet
- `in_ch::Int`: input channel size 
- `out_ch::Int`: output channel size
- `block_fn`: Number of block layers will be generated for this group
- `cifar_stem::Bool`: True if input imagesize <= 32 as CIFAR to use different architecture.

Returns:
- Sequential layers of ResNet model
"""
function ResNet_model(layers::Array{Int, 1}, in_ch::Int, out_ch::Int, block_fn, cifar_stem::Bool)    
    block_fn==BottleneckBlock ? expansion=4 : expansion=1
    if cifar_stem
        in_model = [
        Conv2d(3, in_ch, 64, stride=1, pad=1, bias=false),
        BatchNorm2d(64, relu)
        ]
    else
        in_model = [
        Conv2d(7, in_ch, 64, stride=2, pad=1, bias=false),
        BatchNorm2d(64, relu),
        Pool2d(3, stride=2, pad=1)
        ]
    end

    block_groups = [
        block_group(block_fn, 64, 64, layers[1], stride=1),
        block_group(block_fn, expansion*64, 128, layers[2], stride=2),
        block_group(block_fn, expansion*128, 256, layers[3], stride=2),
        block_group(block_fn, expansion*256, 512, layers[4], stride=2)
    ]  

    out_model = [
        GlobAvgPool2d,
        flatten
        #Linear(expansion*512, out_ch)
        ]


    model_layers = vcat(in_model, block_groups, out_model)
    return Sequential(model_layers)
end

"""
ResNet Generator

Select ResNet architecture to generate.

Arguments:
- `depth::Int`: Total number of layers for ResNet
- `in_ch::Int`: input channel size 
- `out_ch::Int`: output channel size
- `cifar_stem::Bool`: True if input imagesize <= 32 as CIFAR to use different architecture.

Returns:
- ResNet Model

"""
function ResNet(;depth::Int, in_ch::Int=3, out_ch::Int=10, cifar_stem::Bool )
    if depth == 18
        model = ResNet_model([2, 2, 2, 2], in_ch, out_ch, ResidualBlock, cifar_stem)
    elseif depth == 34
        model = ResNet_model([3, 4, 6, 3], in_ch, out_ch, ResidualBlock, cifar_stem)
    elseif depth == 50
        model = ResNet_model([3, 4, 6, 3], in_ch, out_ch, BottleneckBlock, cifar_stem)
    elseif depth == 101
        model = ResNet_model([3, 4, 23, 3], in_ch, out_ch, BottleneckBlock, cifar_stem)
    elseif depth == 152
        model = ResNet_model([3, 8, 36, 3], in_ch, out_ch, BottleneckBlock, cifar_stem)
    else
        error("Not a valid resnet_depth: ", depth)
    end
    model
end


