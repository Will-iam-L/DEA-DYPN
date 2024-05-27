import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.spatial.distance as dis


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class Encoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    # 即embedding:一维卷积层
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)  # 卷积核尺寸为1*1，深度为in_channels=节点数量

    def forward(self, input):
        output = self.conv(input)
        return output  # (batch, hidden_size, seq_len)


class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 3 * hidden_size),
                                          device=device, requires_grad=True))

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden):
        batch_size, hidden_size, _ = static_hidden.size()

        hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)
        # 如果dynamic_hidden维度和static_hidden维度不一样，将dynamic_hidden的维度拓展为static_hidden的维度，以便进行拼接
        hidden = torch.cat((static_hidden, dynamic_hidden, hidden), 1)

        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        attns = F.softmax(attns.clone(), dim=2)  # (batch, seq_len)
        return attns


class Pointer(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, hidden_size, num_layers=1, dropout=0.2):
        super(Pointer, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size),
                                          device=device, requires_grad=True))

        # Used to compute a representation of the current decoder output
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.encoder_attn = Attention(hidden_size)

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh):
        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)
        rnn_out = rnn_out.squeeze(1)

        # Always apply dropout on the RNN output
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh)

            # Given a summary of the output, find an  input context
        enc_attn = self.encoder_attn(static_hidden, dynamic_hidden, rnn_out)  # attention
        context = enc_attn.bmm(static_hidden.permute(0, 2, 1))  # (B, 1, num_feats)

        # Calculate the next output using Batch-matrix-multiply ops
        context = context.transpose(1, 2).expand_as(static_hidden)
        energy = torch.cat((static_hidden, context), dim=1)  # (B, num_feats, seq_len)

        v = self.v.expand(static_hidden.size(0), -1, -1)
        W = self.W.expand(static_hidden.size(0), -1, -1)

        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)

        return probs, last_hh


class DRL4TSP(nn.Module):
    """Defines the main Encoder, Decoder, and Pointer combinatorial models.

    Parameters
    ----------
    static_size: int
        Defines how many features are in the static elements of the model
        (e.g. 2 for (x, y) coordinates)
    dynamic_size: int > 1
        Defines how many features are in the dynamic elements of the model
        (e.g. 2 for the VRP which has (load, demand) attributes. The TSP doesn't
        have dynamic elements, but to ensure compatility with other optimization
        problems, assume we just pass in a vector of zeros.
    hidden_size: int
        Defines the number of units in the hidden layer for all static, dynamic,
        and decoder output units.
    update_fn: function or None
        If provided, this method is used to calculate how the input dynamic
        elements are updated, and is called after each 'point' to the input element.
    mask_fn: function or None
        Allows us to specify which elements of the input sequence are allowed to
        be selected. This is useful for speeding up training of the networks,
        by providing a sort of 'rules' guidlines to the algorithm. If no mask
        is provided, we terminate the search after a fixed number of iterations
        to avoid tours that stretch forever
    num_layers: int
        Specifies the number of hidden layers to use in the decoder RNN
    dropout: float
        Defines the dropout rate for the decoder
    """

    def __init__(self, static_size, dynamic_size, hidden_size,
                 update_fn=None, mask_fn=None, num_layers=1, dropout=0.):
        super(DRL4TSP, self).__init__()

        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')

        self.update_fn = update_fn
        self.mask_fn = mask_fn

        # Define the encoder & decoder models
        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.decoder = Encoder(static_size, hidden_size)
        self.pointer = Pointer(hidden_size, num_layers, dropout)
        # self.Conv1d = Conv1d(hidden_size)


        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        # Used as a proxy initial state in the decoder when not specified
        self.x0 = torch.zeros((1, static_size, 1), requires_grad=True, device=device)

    def forward(self, static, dynamic, decoder_input=None, if_test=False, last_hh=None):
        """
        Parameters
        ----------
        static: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates, which won't change
        dynamic: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the VRP, this can be
            things like the (load, demand) of each city. If there are no dynamic
            elements, this can be set to None
        decoder_input: Array of size (batch_size, num_feats)
            Defines the outputs for the decoder. Currently, we just use the
            static elements (e.g. (x, y) coordinates), but this can technically
            be other things as well
        last_hh: Array of size (batch_size, num_hidden)
            Defines the last hidden state for the RNN
        """
        if if_test:  # 只用在kp_solver里面
            batch_size, input_size, sequence_size = static.size()
            # 扩展 static 张量，以便可以计算两两之间的距离
            static_expanded = static.unsqueeze(2).expand(batch_size, input_size, sequence_size, sequence_size)
            # 计算差异的平方
            diff_square = (static_expanded - static_expanded.transpose(2, 3)) ** 2
            # 对特征维度进行求和，得到距离的平方
            dist_square = diff_square.sum(1)
            # 开根号得到欧氏距离
            dis_matrix = torch.sqrt(dist_square)

            if decoder_input is None:
                decoder_input = self.x0.expand(batch_size, -1, -1)

            # Always use a mask - if no function is provided, we don't update it
            mask = torch.ones(batch_size, sequence_size, device=device)
            # 针对static的每一行i，对于大于1的数j,如果static[i,:,j]与static[i,:,0]相同，则标记mask[i,j]=0
            # 通过扩展第一列和所有列，然后比较来创建一个掩码
            first_column = static[:, :, 0].unsqueeze(2)  # 扩展第一列以便广播
            mask_update = torch.all(static == first_column, dim=1)  # 比较所有列与第一列
            # 更新 mask：如果 mask_update 中的元素为 True，则对应的 mask 元素设置为 0
            mask[mask_update] = 0
            mask[:, 0] = 1

            # Structures for holding the output sequences
            tour_idx, tour_logp = [], []
            max_steps = sequence_size if self.mask_fn is None else 1000
            # 如果mask_fn存在，那迭代的终止条件是mask每个元素都为0（每个城市都已经被访问过了），否则设为一个很大的值1000

            # Static elements only need to be processed once, and can be used across
            # all 'pointing' iterations. When / if the dynamic elements change,
            # their representations will need to get calculated again.
            static_hidden = self.static_encoder(static)
            dynamic_hidden = self.dynamic_encoder(dynamic)

            for i in range(max_steps):

                if not mask.byte().any():  # tensor.any()功能: 如果张量tensor中存在一个元素为True, 那么返回True; 只有所有元素都是False时才返回False，即若mask每个元素都置0时，程序终止
                    break

                # ... but compute a hidden rep for each element added to sequence
                decoder_hidden = self.decoder(decoder_input)  # 做一个映射（256,2,1）->（256,128,1）

                probs, last_hh = self.pointer(static_hidden,
                                              dynamic_hidden,
                                              decoder_hidden, last_hh)

                probs = F.softmax(probs + mask.log(), dim=1).clone()  # 这里不用clone的话会导致一个inplace的报错

                # When training, sample the next step according to its probability.
                # During testing, we can take the greedy approach and choose highest
                if self.training:  # training阶段的decoding方式（sampling）
                    # if True:
                    m = torch.distributions.Categorical(probs)
                    ptr = m.sample()
                    while not torch.gather(mask, 1,
                                           ptr.data.unsqueeze(1)).byte().all():  # gather函数是索引函数，第二个输入是dim=1,第三个参数是索引号
                        print("\n注意！！！出现了坏的采样\n")
                        print("ptr=", ptr)
                        print("mask=", mask)
                        ptr = m.sample()
                    logp = m.log_prob(ptr)
                else:
                    prob, ptr = torch.max(probs, 1)  # Greedy
                    logp = prob.log()  # 概率取log

                # After visiting a node update the dynamic representation
                if self.update_fn is not None:
                    dynamic = self.update_fn(dis_matrix, ptr.data)
                    dynamic_hidden = self.dynamic_encoder(dynamic)

                # And update the mask so we don't re-visit if we don't need to
                if self.mask_fn is not None:  # 更新mask
                    mask = self.mask_fn(mask, dynamic, ptr.data).detach()

                tour_logp.append(logp.unsqueeze(1))  # 每次point会生成一个logp，然后加入tour_logp
                tour_idx.append(ptr.data.unsqueeze(1))  #

                # 这里更改decoder_input计算方法，除了考虑上一个点的坐标，还要考虑起点和终点坐标

                decoder_input = torch.gather(static, 2,
                                             ptr.view(-1, 1, 1)
                                             .expand(-1, input_size, 1)).detach()  # 上一个点

            tour_idx = torch.cat(tour_idx.copy(), dim=1)  # (batch_size, seq_len)
            tour_logp = torch.cat(tour_logp.copy(), dim=1)  # 更改形状：(batch_size, seq_len)
            # tour_logp用来计算梯度的
        else:
            batch_size, input_size, sequence_size = static.size()

            # 根据static计算距离矩阵
            dis_matrix = np.zeros((batch_size, sequence_size, sequence_size))

            for i in range(batch_size):
                # 距离矩阵
                dis_matrix[i, :, :] = dis.cdist(static[i, :, :].cpu().T, static[i, :, :].cpu().T, metric='euclidean')
            dis_matrix = torch.from_numpy(dis_matrix).to(device).float()
            if decoder_input is None:
                decoder_input = self.x0.expand(batch_size, -1, -1)

            # Always use a mask - if no function is provided, we don't update it
            mask = torch.ones(batch_size, sequence_size, device=device)


            # Structures for holding the output sequences
            tour_idx, tour_logp = [], []
            max_steps = sequence_size if self.mask_fn is None else 1000
            # 如果mask_fn存在，那迭代的终止条件是mask每个元素都为0（每个城市都已经被访问过了），否则设为一个很大的值1000

            # Static elements only need to be processed once, and can be used across
            # all 'pointing' iterations. When / if the dynamic elements change,
            # their representations will need to get calculated again.
            static_hidden = self.static_encoder(static)
            dynamic_hidden = self.dynamic_encoder(dynamic)

            for i in range(max_steps):

                if not mask.byte().any():  # tensor.any()功能: 如果张量tensor中存在一个元素为True, 那么返回True; 只有所有元素都是False时才返回False，即若mask每个元素都置0时，程序终止
                    break

                # ... but compute a hidden rep for each element added to sequence
                decoder_hidden = self.decoder(decoder_input)  # 做一个映射（256,2,1）->（256,128,1）
                probs, last_hh = self.pointer(static_hidden,
                                              dynamic_hidden,
                                              decoder_hidden, last_hh)
                probs = F.softmax(probs + mask.log(), dim=1).clone()

                # When training, sample the next step according to its probability.
                # During testing, we can take the greedy approach and choose highest
                if self.training:   # training阶段的decoding方式（sampling）
                # if True:
                    m = torch.distributions.Categorical(probs)
                    ptr = m.sample()
                    while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():  # gather函数是索引函数，第二个输入是dim=1,第三个参数是索引号
                        print("\n注意！！！出现了坏的采样\n")
                        print("ptr=", ptr)
                        print("mask=", mask)
                        ptr = m.sample()
                    logp = m.log_prob(ptr)
                else:
                    prob, ptr = torch.max(probs, 1)  # Greedy
                    logp = prob.log()  # 概率取log

                # After visiting a node update the dynamic representation
                if self.update_fn is not None:
                    dynamic = self.update_fn(dis_matrix, ptr.data)
                    dynamic_hidden = self.dynamic_encoder(dynamic)

                # And update the mask so we don't re-visit if we don't need to
                if self.mask_fn is not None:  # 更新mask
                    mask = self.mask_fn(mask, dynamic, ptr.data).detach()

                tour_logp.append(logp.unsqueeze(1))  # 每次point会生成一个logp，然后加入tour_logp
                tour_idx.append(ptr.data.unsqueeze(1))  #

                # 这里更改decoder_input计算方法，除了考虑上一个点的坐标，还要考虑起点和终点坐标
                decoder_input = torch.gather(static, 2,
                                              ptr.view(-1, 1, 1)
                                              .expand(-1, input_size, 1)).detach()  # 上一个点

            tour_idx = torch.cat(tour_idx.copy(), dim=1) # (batch_size, seq_len)
            tour_logp = torch.cat(tour_logp.copy(), dim=1)  # 更改形状：(batch_size, seq_len)
            # tour_logp用来计算梯度的
        return tour_idx, tour_logp

if __name__ == '__main__':
    raise Exception('Cannot be called from main')
