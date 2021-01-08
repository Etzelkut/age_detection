from depen import *
from performer import default

#copied from https://fast-transformers.github.io/attention/
#https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/attention_layer.py
#updated

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(nn.Module):

    def __init__(self, query_dimensions, feature_map=None, eps=1e-6,):
        super(LinearAttention, self).__init__()
        self.feature_map = feature_map or elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, context_mask):
        # Apply the feature map to the queries and keys
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        K = K * context_mask.float()[:, :, None, None]

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("nshd,nshm->nhmd", K, values)

        # Compute the normalizer
        Z = 1/(torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))+self.eps)

        # Finally compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)

        return V.contiguous()



class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None,
                 d_values=None, feature_map = None):
        super(AttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = LinearAttention(d_model, feature_map)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        print("linear")

    def forward(self, queries, context = None, mask = None, context_mask = None):

        context = default(context, queries)
        context_mask = default(context_mask, mask)
        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = context.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(context).view(N, S, H, -1)
        values = self.value_projection(context).view(N, S, H, -1)

        # Let the world know of the qkv

        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            context_mask
        ).view(N, L, -1)

        # Project the output and return
        return self.out_projection(new_values)