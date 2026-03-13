import torch
import torch.nn as nn
from einops import rearrange, einsum

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        #Construct an embedding module. This function should accept the following parameters
        
        std = 1
        empty_embedding = torch.empty([num_embeddings,embedding_dim],device=device,dtype=dtype)
        self.weight = nn.Parameter(empty_embedding)
        nn.init.trunc_normal_(self.weight,mean=0,std=std,a=-3*std,b=3*std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor :
        #Lookup the embedding vectors for the given token IDs.
        #embedding: vocab_size , d_model; token_ids:batch seq -> batch seq d_model
        #即embedding是字典（二维），用tokenids去查字典
        return self.weight[token_ids]

if __name__ == "__main__":
    pass