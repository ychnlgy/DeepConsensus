import torch
import cnn_example, deepconsensus

class DeepConsensusCnn(cnn_example.Cnn):
    
    def __init__(self, channels, classes, imagesize):
        super(DeepConsensusCnn, self).__init__(channels, classes, imagesize)
        
        self.deepconsensuslayers = torch.nn.ModuleList([
            
            # for Cnn BLOCK 2
            deepconsensus.DeepConsensusLayer(64, classes),
            
            # for Cnn BLOCK 4
            deepconsensus.DeepConsensusLayer(128, classes),
            
            # for Cnn BLOCK 6
            deepconsensus.DeepConsensusLayer(256, classes)
            
        ])
        
        # Maps Cnn BLOCK number to DeepConsensusLayer
        self.deepconsensus_map = {
            2: self.deepconsensuslayers[0],
            4: self.deepconsensuslayers[1],
            6: self.deepconsensuslayers[2]
        }
        
        # remove inherited fields that are not useful.
        del self.avg
        del self.fc
    
    def forward(self, X):
        consensus = []
        for i, layer in enumerate(self.net):
            X = layer(X)
            if i in self.deepconsensus_map:
                dc_layer = self.deepconsensus_map[i]
                consensus.append(dc_layer(X))
        return sum(consensus)
