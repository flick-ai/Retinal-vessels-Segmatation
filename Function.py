def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)