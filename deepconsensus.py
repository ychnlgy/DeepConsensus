import torch

class DeepConsensusLayer(torch.nn.Module):

    def __init__(self, hiddensize, classes):
        super(DeepConsensusLayer, self).__init__()
        self.h = torch.nn.Sequential(
            torch.nn.Conv2d(hiddensize, hiddensize, 1),
            torch.nn.LeakyReLU()
        )
        self.prototypes = torch.nn.Parameter(
            torch.rand(classes, hiddensize).normal_(mean=0, std=0.02)
        )
        self.cos = torch.nn.CosineSimilarity(dim=1)
    
    def forward(self, layer_output):
        N, C, W, H = layer_output.size()
        summary = self.h(layer_output).view(N, -1, W*H).sum(dim=-1)
        return self.classify(summary)
    
    def classify(self, summary):
        prototypes = self.prototypes/self.prototypes.norm(dim=1).view(-1, 1)
        
        N, D = summary.size()
        C, D = prototypes.size()
        summary = summary.view(N, 1, D).repeat(1, C, 1).view(N*C, D)
        prototypes = prototypes.repeat(N, 1)
        
        matched = self.cos(summary, prototypes)
        return matched.view(N, C)
