import torch
import torch.nn as nn
import torch.nn.functional as F
def _default_init_func(_m):
    if isinstance(_m, nn.Linear):
        nn.init.trunc_normal_(_m.weight, std=.02)
        if _m.bias is not None:
            nn.init.constant_(_m.bias, 0)
    elif isinstance(_m, nn.LayerNorm):
        nn.init.constant_(_m.bias, 0)
        nn.init.constant_(_m.weight, 1.0)
    elif isinstance(_m, nn.Parameter):
        nn.init.trunc_normal_(_m, std=.02)
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.gelu
    def forward(self, query, value):
        query2 = self.norm1(query)
        query2 = self.self_attn(query2, query2, value=query2,need_weights=False)[0]
        query = query + self.dropout1(query2)
        query2 = self.norm2(query)
        query2 = self.multihead_attn(query=query2,
                                     key=value,
                                     value=value, need_weights=False)[0]
        query = query + self.dropout2(query2)
        query2 = self.norm3(query)
        query2 = self.linear2(self.dropout(self.activation(self.linear1(query2))))
        query = query + self.dropout3(query2)
        return query
class WeightsGenerater(nn.Module):
    def __init__(self, dim, depth, \
                 llm_dim, weights_num, lora_rank, \
                 mlp_ratio=4, num_heads=16):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, weights_num, dim), requires_grad=True)
        self.decoders = nn.ModuleList([TransformerDecoderLayer(dim, num_heads, int(dim*mlp_ratio)) for i in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.weights_head = nn.Linear(dim, llm_dim*lora_rank)
    def forward(self, img_feature):
        query = self.queries.expand(img_feature.shape[0], -1, -1)
        for layer in self.decoders:
            query = layer(query, img_feature)
        query = self.weights_head(self.norm(query))
        return query  # [batch_size, weights_num, llm_dim*lora_rank]
class LoRAGenerater(nn.Module):
    def __init__(self, dim, depth, \
                 visual_dim, pos_num, \
                 llm_dim, llm_depth, lora_rank, \
                 lora_type='qkvom', weights_sep=True, \
                 mlp_ratio=4, num_heads=16, skip_layers=1, vlora_alpha=None):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, pos_num, dim), requires_grad=True)
        self.lora_type = lora_type
        self.weights_sep = weights_sep
        self.skip_layers = skip_layers
        if self.weights_sep:
            self.gen = nn.ModuleList([WeightsGenerater(dim, depth, llm_dim, llm_depth//skip_layers, lora_rank, mlp_ratio, num_heads) for i in range(len(lora_type))])
        else:
            self.gen = WeightsGenerater(dim, depth, llm_dim, len(lora_type)*llm_depth//skip_layers, lora_rank, mlp_ratio, num_heads)
        # initialization before init self.Bs
        self.apply(_default_init_func)
        self.Bs = nn.Parameter(torch.zeros(1, len(lora_type)*llm_depth//skip_layers, llm_dim*lora_rank), requires_grad=True)
        self.vlora_alpha = vlora_alpha if vlora_alpha is not None else lora_rank
        self.llm_depth = llm_depth
        self.llm_dim = llm_dim
        self.lora_rank = lora_rank
    def forward(self, img_feature):
        """
        img_features: output by vision encoder, shape of [batch_size, pos_num, visual_dim]
        output: [{'q': (A, B), ...},...] A's shape: [batch_size, llm_dim, lora_rank]
                                   B's shape: [1, llm_dim, lora_rank]
                len(output)=llm_depth
        """
        img_feature = self.visual_proj(img_feature) + self.pos_embed
        if self.weights_sep:
            weights = []
            for sub_gen in self.gen:
                weights.append(sub_gen(img_feature))
            weights = torch.cat(weights, dim=1)  # [batch_size, len(lora_type)*llm_depth, llm_dim*lora_rank]
        else:
            weights = self.gen(img_feature)  # [batch_size, len(lora_type)*llm_depth, llm_dim*lora_rank]
        weights = weights.reshape(weights.shape[0], len(self.lora_type), self.llm_depth//self.skip_layers, self.llm_dim, self.lora_rank)
        Bs = self.Bs.reshape(1, len(self.lora_type), self.llm_depth//self.skip_layers, self.llm_dim, self.lora_rank)
        Bs = self.vlora_alpha / self.lora_rank * Bs
        lora_weights_list = []
        for depth in range(self.llm_depth):
            lora_weights = {}
            # follow flamingo, not insert cross_layer before the first layer
            if((depth+1) % self.skip_layers == 0):
                for i, type in enumerate(str(self.lora_type)):
                    A = weights[:, i, depth//self.skip_layers]
                    B = Bs[:, i, depth//self.skip_layers]
                    lora_weights[type] = (A, B)
                for j in ['q', 'k', 'v', 'o', 'm']:
                    if j not in lora_weights:
                        lora_weights[j] = (None, None)
            else:
                for j in ['q', 'k', 'v', 'o', 'm']:
                    lora_weights[j] = (None, None)
            lora_weights_list.append(lora_weights)
        return lora_weights_list
def get_lora_generater(model_args):
    return LoRAGenerater(
            dim = model_args.vlora_dim,
            depth = model_args.vlora_depth,
            visual_dim = model_args.vlora_visual_dim,
            pos_num = model_args.vlora_pos_num,
            llm_dim = model_args.vlora_llm_dim,
            llm_depth = model_args.vlora_llm_depth,
            lora_rank = model_args.vlora_rank,
            lora_type = model_args.vlora_type,
            weights_sep = model_args.weights_sep,
            skip_layers = model_args.skip_layers,
            vlora_alpha = model_args.vlora_alpha,
    )
if __name__ == "__main__":
    model = get_lora_generater('llava-v1.5-7b').cuda()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameter nums: {total_params}')
    img_feature = torch.randn(3, 576, 1024).cuda()
    lora_weights = model(img_feature)
    print(len(lora_weights))
    lora_weights_sub = lora_weights[0]
    for k in lora_weights_sub:
        print(k, lora_weights_sub[k][0].shape, lora_weights_sub[k][1].shape)
