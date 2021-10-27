import torch
from torch import nn

class StrScorer(nn.Module):
    def __init__(self, architecture=[512,512,512], lr=1e-4, n=4, loss=nn.BCEWithLogitsLoss(reduction='none'),
            embed_dim=32, topk_count=10, cuda=True):
        super().__init__()

        self.local_layers = nn.ModuleList()
        self.lr = lr
        self.topk_count = topk_count

        # embed bytes into an embed_dim space
        self.embed = nn.Embedding(257, embed_dim, max_norm=1, padding_idx=0)

        # several layers of conv1d followed by prelu activations.
        # use weight norm instead of batch norm. we're doing weird things that will
        # throw off batchnorm.
        prev_h = embed_dim
        for h in architecture:
            self.local_layers.append(nn.utils.weight_norm(nn.Conv1d(prev_h, h, n)))
            self.local_layers.append(nn.PReLU(h))

            prev_h = h

        self.out_layer = nn.Conv1d(prev_h, 1, 1)

        self.loss = loss
        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.sched = torch.optim.lr_scheduler.StepLR(self.opt, 1000, 0.95)

        self.use_cuda = cuda
        if self.use_cuda:
            self.cuda()

        print(self)

    def forward(self, x):
        if self.use_cuda:
            x = x.cuda()

        # embed and transpose for the conv1d layer
        x = self.embed(x)
        x = x.transpose(2,1)

        # feed through conv/prelu layers
        for l in self.local_layers:
            x = l(x)

        # get outputs for model as logits
        y = self.out_layer(x)

        return y

    def fit(self, x, y):
        self.train()
        self.opt.zero_grad()

        y = y.reshape(-1,1)

        if self.use_cuda:
            x = x.cuda()
            y = y.cuda()

        # get activations
        yp = self.forward(x)
        y_mask = (x == 0).float()[:,yp.shape[2]]

        # squeeze it
        yp = yp.squeeze()

        # retain top-k scores; get the average of these scores.
        yp_topk = torch.topk(yp, self.topk_count, dim=1)[0]
        yp_max = yp_topk[:,0]

        loss = self.loss(yp_topk, y.repeat(1, yp_topk.shape[1])).mean()

        # backprop
        loss.backward()

        # update grads
        self.opt.step()

        self.sched.step()

        # return predictions and loss as np arrays
        return yp_max.detach().cpu().numpy(), loss.detach().cpu().numpy()
