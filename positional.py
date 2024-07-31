import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(seq_len, d_model):
    positions = np.arange(seq_len)[:, np.newaxis]
    angles = np.arange(d_model)[np.newaxis, :] / d_model
    angles = positions / np.power(10000, angles)
    
    encoding = np.zeros((seq_len, d_model))
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])
    
    return encoding

seq_len = 100
d_model = 512

pe = positional_encoding(seq_len, d_model)

plt.figure(figsize=(10, 5))
plt.imshow(pe, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Positional Encoding')
plt.xlabel('Dimension')
plt.ylabel('Position')
plt.show()

# Check if the encoding is the same for a given position across different dimensions
print("Encoding for position 10:")
print(pe[10])

# Check if the encoding changes smoothly across positions
print("\nEncoding difference between positions 10 and 11:")
print(pe[11] - pe[10])

# Verify that the encoding can handle longer sequences
long_seq = positional_encoding(1000, d_model)
print("\nShape of encoding for a longer sequence:", long_seq.shape)

def get_relative_position_encoding(pos1, pos2, d_model):
    return pe[pos1] - pe[pos2]

# Example: Get relative encoding between positions 5 and 10
relative_encoding = get_relative_position_encoding(5, 10, d_model)
print("\nRelative encoding between positions 5 and 10:")
print(relative_encoding)