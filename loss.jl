
using Knet
using LinearAlgebra

function loss_contrastive(model, x, bs)

    LARGE_NUM = 1e9
    temperature = 0.5
    
    z = transpose(model(x))
    
    _atype = KnetArray{Float32}    
    norm_z = sqrt.(sum(abs2,z,dims=2))

    zx = z ./ norm_z

    z1 = zx[1:bs,:]
    z2 = zx[bs+1:bs*2,:]
    
    
    labels = convert(_atype, Array{Float64}(I, bs, bs*2))
    mask = convert(_atype, Array{Float64}(I, bs, bs)*LARGE_NUM)
    
    logits_aa = z1*transpose(z1)/temperature - mask 
    logits_bb = z2*transpose(z2)/temperature - mask 
    logits_ab = z1*transpose(z2)/temperature
    logits_ba = z2*transpose(z1)/temperature
    
    loss_a = sum(-labels.*logsoftmax([logits_ab logits_aa], dims=2))/bs
    loss_b = sum(-labels.*logsoftmax([logits_ba logits_bb], dims=2))/bs
    
    loss = loss_a + loss_b

    return loss
end
