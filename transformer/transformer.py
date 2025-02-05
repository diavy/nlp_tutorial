import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.QM = nn.Linear(hidden_size, hidden_size, bias = False)
        self.KM = nn.Linear(hidden_size, hidden_size, bias = False)
        self.VM = nn.Linear(hidden_size, hidden_size, bias = False)

    def forward(
            self,
            query,
            key,
            value,
            mask = None
    ):
        # It is very true that query, key, value are duplicates of the input, however, this is True only in encoder. In decoder, the attetnion class could also be used, but the difference is that the query is different from key, value.

        # query [batch_size, q_seq_length, hidden_size]
        query = self.QM(query)
        # query [batch_size, q_seq_length, hidden_size]

        # key [batch_size, k_seq_length, hidden_size]
        key = self.KM(key)
        # key [batch_size, k_seq_length, hidden_size]

        # value [batch_size, v_seq_length, hidden_size]
        value = self.VM(value)
        # value [batch_size, v_seq_length, hidden_size]

        # Keep track of the size.
        QK_prod = torch.einsum('bqh,bkh->bqk', query, key)
        # word similarities
        if mask is not None:
            #QK_prod = QK_prod.masked_fill(mask == 0, float('-1e20'))
            QK_prod = QK_prod + mask
        attention = torch.softmax(QK_prod / (self.hidden_size ** 0.5), dim = 2)
        # attention,[batch_size, query_length, key_length]
        output = torch.einsum('bqk,bkh->bqh', attention, value)
        return output


class TransformerBlock(nn.Module):
    def __init__(
            self,
            hidden_size,
            dropout,
            forward_expansion
    ):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(
                hidden_size,
                forward_expansion * hidden_size
            ),
            nn.ReLU(),
            nn.Linear(
                forward_expansion * hidden_size, hidden_size
            )
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask = None):
        attention = self.attention(query, key, value, mask)
        output = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(output)
        block_output = self.dropout(self.norm2(forward + output))
        return block_output


class Encoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            hidden_size,
            num_layers,
            forward_expansion,
            dropout
    ):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, dropout, forward_expansion)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask = None):
        hidden_inputs = self.embedding(x)
        for layer in self.layers:
            hidden_inputs = layer(
                query = hidden_inputs,
                key = hidden_inputs,
                value = hidden_inputs,
                mask = mask
            )
        return hidden_inputs


class DecoderBlock(nn.Module):
    def __init__(
            self,
            hidden_size,
            forward_expansion,
            dropout
    ):
        super(DecoderBlock, self).__init__()
        self.maskedattention = SelfAttention(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.transformer_block = TransformerBlock(
            hidden_size,
            dropout,
            forward_expansion
        )

    def forward(self, query, key, value, mask):
        # For decoder, the case is different
        # For each decoder block, query is directory from decoder input.However the key and value are duplicates of the enocder output
        # Therefore, for the masked_attention, the input is query = query, key = query, value = query
        masked_attention_query = self.maskedattention(query, query, query, mask)
        masked_attention_query = self.dropout(self.norm(masked_attention_query + query))
        decoder_block_output = self.transformer_block(
            query = masked_attention_query,
            key = key,
            value = value,
            mask = None
        )
        return decoder_block_output


class Decoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            hidden_size,
            num_layers,
            forward_expansion,
            dropout
    ):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [DecoderBlock(hidden_size, forward_expansion, dropout)
             for _ in range(num_layers)]
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, decoder_x, encoder_output, mask):
        decoder_hidden_input = self.embedding(decoder_x)
        for layer in self.layers:
            decoder_hidden_input = layer(
                query = decoder_hidden_input,
                key = encoder_output,
                value = encoder_output,
                mask = mask
            )
        output = self.fc(decoder_hidden_input)
        return output


class Transformer(nn.Module):
    def __init__(
            self,
            hidden_size = 512,
            encoder_vocab_size = 20000,
            decoder_vocab_size = 20000,
            num_encoder_layers = 3,
            num_decoder_layers = 3,
            forward_expansion = 4,
            dropout = 0.1
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            vocab_size = encoder_vocab_size,
            hidden_size = hidden_size,
            num_layers = num_encoder_layers,
            forward_expansion = forward_expansion,
            dropout = dropout
        )
        self.decoder = Decoder(
            vocab_size = decoder_vocab_size,
            hidden_size = hidden_size,
            num_layers = num_decoder_layers,
            forward_expansion = forward_expansion,
            dropout = dropout
        )

    def forward(self, encoder_input_ids, decoder_input_ids):
        decoder_seqlen = decoder_input_ids.shape[1]
        mask = generate_square_subsequent_mask(s = decoder_seqlen)
        print(mask)
        encoder_output = self.encoder(encoder_input_ids)
        out = self.decoder(decoder_input_ids, encoder_output, mask)
        # batch_size, seq_len, vocab_size
        return out


def generate_square_subsequent_mask(s):
    mask = torch.tril(torch.ones(s, s))
    mask = torch.where(
        mask == 1.0,
        torch.Tensor([0.0]),
        torch.Tensor([float('-inf')])
    )
    return mask


if __name__ == '__main__':
    import pickle
    batch = pickle.load(open('data/tmp_batch.pkl', 'rb'))

    # encoder = Encoder(vocab_size=20000,hidden_size=512,num_layers=3,forward_expansion=4, dropout=0.1)

    x = batch['input_ids']
    model = Transformer()
    out = model(x, x)
    print(out.shape)

    print('end')