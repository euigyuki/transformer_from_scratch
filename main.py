from model import Transformer
import pandas as pd
from data import data
import torch

#testing main

def main():

    max_len = 5000
    d_model =2
    df = pd.DataFrame(data)
    df.to_csv('translation_data.csv', index=False)
    position = torch.arange(max_len).unsqueeze(1)
    print(position.shape)
    print(position)
    print()

    pe = torch.zeros(max_len, 1, d_model)
    print(pe.shape)
    print(pe)
    # # Example usage
    # src_vocab_size = 5000
    # tgt_vocab_size = 5000
    # d_model = 512
    # num_heads = 8
    # num_layers = 6
    # d_ff = 2048
    # max_seq_length = 100
    # dropout = 0.1

    # model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

    # # Generate dummy input
    # src = torch.randint(1, src_vocab_size, (64, 10))  # (batch_size, src_seq_len)
    # tgt = torch.randint(1, tgt_vocab_size, (64, 12))  # (batch_size, tgt_seq_len)
    # src_mask = torch.ones(64, 1, 10)  # (batch_size, 1, src_seq_len)
    # tgt_mask = torch.tril(torch.ones(64, 12, 12))  # (batch_size, tgt_seq_len, tgt_seq_len)

    # output = model(src, tgt, src_mask, tgt_mask)
    # print(output.shape)  # Expected: (64, 12, tgt_vocab_size)

if __name__ == '__main__':
    main()