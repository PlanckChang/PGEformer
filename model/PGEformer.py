import torch.nn as nn
import torch

class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(model_dim, num_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.GELU(),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, time_step, model_dim) or (batch_size, num_nodes, model_dim)
        residual = x
        out, _ = self.attn(x, x, x, need_weights=False)
        out = self.dropout1(out)
        out = self.ln1(residual + out)
        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)
        return out


class PGEformer(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=256,
        tod_embedding_dim=128,
        dow_embedding_dim=128,
        spatial_embedding_dim=256,
        feed_forward_dim=1024,
        num_heads=8,
        num_layers=1,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim  # 3  raw data, tod, dow
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.noise_embedding_dim = input_embedding_dim

        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
        )
        
        self.num_heads = num_heads
        self.num_layers = num_layers
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim, _weight=nn.init.xavier_normal_(nn.Parameter(torch.empty(steps_per_day, tod_embedding_dim))))
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim, _weight=nn.init.xavier_normal_(nn.Parameter(torch.empty(7, dow_embedding_dim))))
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        # self.temporal_embedding = nn.Embedding(steps_per_day * 7, self.input_embedding_dim, _weight=nn.init.uniform_(nn.Parameter(torch.empty(steps_per_day * 7, self.input_embedding_dim))))
        self.s_input = nn.Sequential(
            nn.Linear(self.in_steps, self.input_embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_embedding_dim, self.input_embedding_dim)
        )
        self.s_output = nn.Sequential(
            nn.Linear(self.model_dim, feed_forward_dim),
            nn.GELU(),
            nn.Linear(feed_forward_dim, feed_forward_dim),
            nn.GELU(),
            nn.Linear(feed_forward_dim, self.out_steps)    
        )
        self.attn_layers_s = nn.ModuleList(
            [  
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads)
                for _ in range(num_layers)
            ]
        )
        
    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]
        if self.tod_embedding_dim > 0:
            tod = x[:, 0, :, 1].squeeze(-1) # assident case, the batch size is 1
        if self.dow_embedding_dim > 0:
            dow = x[:, 0, :, 2].squeeze(-1)         
        x   = x[:, :, :, 0].squeeze(-1) # tokenization is equal to squeeze(-1) and transpose(1, 2)
        x_s = x.transpose(1, 2)
        x_s = self.s_input(x_s)    # (batch_size, num_nodes, model_dim) inducing the token number to be the same as the node number and bypassing the traversal of the temporal dimension 
        features = [x_s]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, dow_embedding_dim)
            features.append(dow_emb)  
        if self.spatial_embedding_dim > 0: 
            spatial_emb = self.node_emb.expand(
                batch_size, *self.node_emb.shape
            )
            features.append(spatial_emb)

        # tod_dow = self.temporal_embedding((dow.long() * self.steps_per_day + (tod * self.steps_per_day).long() ))
        # features.append(tod_dow)
        x_s = torch.cat(features, dim=-1).contiguous()  # (batch_size, num_nodes, model_dim)    
        for attn in self.attn_layers_s:     # Spatial Module SM 
            x_s = attn(x_s)
        x_s = self.s_output(x_s)            # MLP prediction head
        out = x_s 
        out =  out.transpose(1, 2) # (batch_size, out_steps, num_nodes)
        return out 
    
if __name__ == "__main__":
    pass
