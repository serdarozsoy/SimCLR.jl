# import Pkg; using Pkg; Pkg.add(["Knet","MLDatasets","PyCall","LinearAlgebra","JLD2","CUDA"])

using CUDA
using Knet
using MLDatasets
using PyCall
include("resnet.jl")
using LinearAlgebra
using JLD2

include("data.jl")
include("data_transfomr.jl")
include("loss.jl")

X_train, y_train = load_cifar_dataset();


model_main = Sequential(ResNet(depth=18, in_ch=3, out_ch=512, cifar_stem=true), Linear(512,512,f=relu), Linear(512,128))


for p in params(model_main)
    p.opt = Adam(lr=0.001)
end


epochs = 1000
train_loss = zeros(epochs);
_atype = KnetArray{Float32}

"""
Train s
"""
for epoch in 1:epochs
    train_loss[epoch] = 0.0
    batch_count = 0
    dtrn = minibatch(X_train, y_train, 512, shuffle=true)
    for (batch_x, batch_y) in dtrn
        bs = size(batch_x)[4]
        batch_x1 = cat([transform(batch_x[:,:,:,i]) for i in 1:bs]..., dims=4)
        batch_x2 = cat([transform(batch_x[:,:,:,i]) for i in 1:bs]..., dims=4) 
        x2 = convert(_atype, permutedims(cat(batch_x1 , batch_x2, dims=4), (2, 3, 1, 4)))
        loss_main= @diff loss_contrastive(model_main, x2, bs)
        train_loss[epoch] += value(loss_main)
        batch_count += 1
        for p in params(model_main)
            g = grad(loss_main, p)
            update!(value(p), g, p.opt)
        end
    end
    train_loss[epoch] /= batch_count
    println("Train epoch: ", epoch ," ,loss: ", train_loss[epoch])
    flush(stdout)
    if mod(epoch, 100) == 0
      model_name = string("results/model_main_", epoch, ".jld2") 
      save(model_name, "model", model_main)
    end
end
