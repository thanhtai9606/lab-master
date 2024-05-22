import torch
import torch.nn as nn
import torch.optim as optim
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
        
    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, max_len=5000):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.embedding_src = nn.Embedding(src_vocab_size, d_model)
        self.embedding_tgt = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        src = self.embedding_src(src) * math.sqrt(self.d_model)
        tgt = self.embedding_tgt(tgt) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt, src_mask, tgt_mask, None, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return self.fc_out(output)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Parameters
src_vocab_size = 10000
tgt_vocab_size = 10000
d_model = 512
nhead = 8
num_encoder_layers = 6  # N=6 layers for encoder
num_decoder_layers = 6  # N=6 layers for decoder
dim_feedforward = 2048
max_len = 5000
batch_size = 32
num_epochs = 10

# Model, loss function, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Transformer(src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_len).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for src, tgt in train_loader:  # Assuming train_loader is defined elsewhere
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:-1, :]
        src_mask = generate_square_subsequent_mask(src.size(0)).to(device)
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(device)
        src_padding_mask = (src == 0).transpose(0, 1)
        tgt_padding_mask = (tgt_input == 0).transpose(0, 1)
        
        optimizer.zero_grad()
        output = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt[1:].reshape(-1))
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Example inference
model.eval()
with torch.no_grad():
    src = torch.randint(1, src_vocab_size, (30, batch_size)).to(device)  # Example source input
    tgt = torch.randint(1, tgt_vocab_size, (30, batch_size)).to(device)  # Example target input
    src_mask = generate_square_subsequent_mask(src.size(0)).to(device)
    tgt_mask = generate_square_subsequent_mask(tgt.size(0)).to(device)
    src_padding_mask = (src == 0).transpose(0, 1)
    tgt_padding_mask = (tgt == 0).transpose(0, 1)
    output = model(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
    print(output)
