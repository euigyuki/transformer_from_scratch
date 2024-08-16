import torch
import torch.nn as nn
import unittest
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        sequence = torch.arange(0,d_model,2).float()
        #div_term = torch.exp(sequence* (-math.log(10000.0) / d_model))
        div_term = 1/torch.pow(10000.0, sequence / d_model)
        pe = torch.zeros(max_len, 1, d_model)
        pe[:,0, 0::2] = torch.sin(position * div_term)
        pe[:,0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        print(x.size(0))
        return x + self.pe[:x.size(0)] # of the maximum lookup table, it will only look at the first x.size(0) elements, which is the length of the input sequence


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads #splitting the embedding dimensions by the number of heads: will focus on specific parts of the embeddings
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        #seq_len is the length of the input sequence, ie. the number of words in the sentence
        #d_model is the dimension of the embedding

        Q = self.q_linear(query) # (batch_size, seq_len, d_model)
        K = self.k_linear(key)   # (batch_size, seq_len, d_model)
        V = self.v_linear(value) # (batch_size, seq_len, d_model)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim) # (batch_size, seq_len, num_heads, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim) # (batch_size, seq_len, num_heads, head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim) # (batch_size, seq_len, num_heads, head_dim)

        Q = Q.transpose(1, 2) # (batch_size, num_heads, seq_len_q, head_dim)
        K = K.transpose(1, 2) # (batch_size, num_heads, seq_len_k, head_dim) 
        V = V.permute(0, 2, 1, 3)  # resulting shape of V is (batch_size, num_heads, seq_len, head_dim)

        K = K.transpose(-2, -1) # (batch_size, num_heads, head_dim, seq_len_k)
        
        energy = torch.matmul(Q, K) / (self.head_dim ** 0.5) #resulting shape is (batch_size, num_heads, seq_len_q, seq_len_k)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy, dim=-1) #resulting shape of attention is (batch_size, num_heads, seq_len_q, seq_len_k)
        x = torch.matmul(attention, V) #resulting shape of x is (batch_size, num_heads, seq_len_q, head_dim)
        cont = x.transpose(1,2).contiguous() #resulting shape of cont is (batch_size, seq_len_q, num_heads, head_dim)
        x = cont.view(batch_size, -1, self.d_model) #resulting shape of x is (batch_size, seq_len_q, d_model)
        return self.out(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Implement encoder layer
        pass

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Implement decoder layer
        pass

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode(self, src, src_mask):
        # Implement encoder
        pass

    def decode(self, tgt, memory, src_mask, tgt_mask):
        # Implement decoder
        pass

    def forward(self, src, tgt, src_mask, tgt_mask):
        # Implement full transformer forward pass
        pass
class TestTransformerComponents(unittest.TestCase):
    def test_positional_encoding(self):
        d_model = 512
        max_len = 5000
        pe = PositionalEncoding(d_model, max_len)
        x = torch.zeros(1, 50, d_model)
        output = pe(x)
        self.assertEqual(output.shape, (1, 50, d_model))

    def test_multi_head_attention(self):
        d_model = 512
        num_heads = 8
        mha = MultiHeadAttention(d_model, num_heads)
        batch_size = 64
        seq_len = 50
        query = key = value = torch.rand(batch_size, seq_len, d_model)
        output = mha(query, key, value)
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

    # def test_feed_forward(self):
    #     d_model = 512
    #     d_ff = 2048
    #     ff = FeedForward(d_model, d_ff)
    #     batch_size = 64
    #     seq_len = 50
    #     x = torch.rand(batch_size, seq_len, d_model)
    #     output = ff(x)
    #     self.assertEqual(output.shape, (batch_size, seq_len, d_model))

    # def test_encoder_layer(self):
    #     d_model = 512
    #     num_heads = 8
    #     d_ff = 2048
    #     dropout = 0.1
    #     encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
    #     batch_size = 64
    #     seq_len = 50
    #     x = torch.rand(batch_size, seq_len, d_model)
    #     mask = torch.ones(batch_size, 1, seq_len)
    #     output = encoder_layer(x, mask)
    #     self.assertEqual(output.shape, (batch_size, seq_len, d_model))

    # def test_decoder_layer(self):
    #     d_model = 512
    #     num_heads = 8
    #     d_ff = 2048
    #     dropout = 0.1
    #     decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
    #     batch_size = 64
    #     seq_len = 50
    #     x = torch.rand(batch_size, seq_len, d_model)
    #     enc_output = torch.rand(batch_size, seq_len, d_model)
    #     src_mask = tgt_mask = torch.ones(batch_size, 1, seq_len)
    #     output = decoder_layer(x, enc_output, src_mask, tgt_mask)
    #     self.assertEqual(output.shape, (batch_size, seq_len, d_model))

    # def test_transformer(self):
    #     src_vocab_size = tgt_vocab_size = 10000
    #     d_model = 512
    #     num_heads = 8
    #     num_layers = 6
    #     d_ff = 2048
    #     max_seq_length = 5000
    #     dropout = 0.1
    #     transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    #     batch_size = 64
    #     src_seq_len = tgt_seq_len = 50
    #     src = torch.randint(src_vocab_size, (batch_size, src_seq_len))
    #     tgt = torch.randint(tgt_vocab_size, (batch_size, tgt_seq_len))
    #     src_mask = tgt_mask = torch.ones(batch_size, 1, 1, src_seq_len)
    #     output = transformer(src, tgt, src_mask, tgt_mask)
    #     self.assertEqual(output.shape, (batch_size, tgt_seq_len, tgt_vocab_size))

if __name__ == '__main__':
    unittest.main()
