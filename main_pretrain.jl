# import Pkg; Pkg.add(["Knet","MLDatasets","PyCall","LinearAlgebra","JLD2","CUDA"])

using CUDA
using Knet
using MLDatasets
using PyCall
include("resnet.jl")
using LinearAlgebra
using JLD2

include("data.jl")
include("data_transform.jl")
include("loss.jl")

X_train, y_train = load_cifar_dataset();

model_main = Sequential(ResNet(depth=18, in_ch=3, out_ch=512, cifar_stem=true), Linear(512,512,f=relu), Linear(512,128))

# Parameters
optim_lr = 0.001
batchsize = 512
epochs = 1000

# Define optimizer
for p in params(model_main)
    p.opt = Adam(lr=optim_lr)
end

train_loss = zeros(epochs);
_atype = KnetArray{Float32}


for epoch in 1:epochs
    train_loss[epoch] = 0.0
    batch_count = 0
    dtrn = minibatch(X_train, y_train, batchsize , shuffle=true)
    for (batch_x, batch_y) in dtrn
        bs = size(batch_x)[4]
        # Transform input according to defined data augmentations
        batch_x1 = cat([transform(batch_x[:,:,:,i]) for i in 1:bs]..., dims=4)
        batch_x2 = cat([transform(batch_x[:,:,:,i]) for i in 1:bs]..., dims=4)
        # Concat batches and permute dimension according to ResNet input dimensions
        batch_x1x2 = convert(_atype, permutedims(cat(batch_x1 , batch_x2, dims=4), (2, 3, 1, 4)))
        # Calculate loss 
        loss_main= @diff loss_contrastive(model_main,b atch_x1x2, bs)
        train_loss[epoch] += value(loss_main)
        batch_count += 1
        # Optimize wall model weights ( encoder + projection head)
        for p in params(model_main)
            g = grad(loss_main, p)
            update!(value(p), g, p.opt)
        end
    end
    train_loss[epoch] /= batch_count
    println("Train epoch: ", epoch ," ,loss: ", train_loss[epoch])
    # In batch running, program waits to finish for printing output
    flush(stdout)
    # Saving model for each 100 epoch
    if mod(epoch, 100) == 0
      model_name = string("results/model_main_", epoch, ".jld2") 
      save(model_name, "model", model_main)
    end
end
