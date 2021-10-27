import time
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import argparse

from model import StrScorer
from utils import softmax, unpackbits, update_buffers, write_stats, RpBuffer, BinaryDataLoader

def setup_model_artifacts(artifactpath, model_name):
    def try_mkdir(dirname):
        try:
            os.mkdir(dirname)
        except:
            pass

    try_mkdir(f"{artifactpath}/")
    try_mkdir(f"{artifactpath}/{model_name}")


if __name__ == "__main__":
    # argparse stuff
    parser = argparse.ArgumentParser(description="Train a signature generation model.")
    parser.add_argument("--uripath", help="path for training sample uris", type=str, required=True)
    parser.add_argument("--cuda", help="use cuda?")
    parser.add_argument("--artifactpath", help="path for tensorboard/model artifacts", type=str, default="model_artifacts")
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument("--kernelsz", help="convolution kernel size", type=int, default=4)
    parser.add_argument("--embed_dim", help="embedding size", type=int, default=16)
    parser.add_argument("--seqlen", help="binary chunk size", type=int, default=16384)
    parser.add_argument("--topk_count", help="number of max's to backprop through", type=int, default=10)
    parser.add_argument("--architecture", help="model architecture", type=str, default="32,64,128,192,256,512")
    args = parser.parse_args()

    uripath = args.uripath
    artifactpath = args.artifactpath
    lr = args.lr
    n = args.kernelsz
    embed_dim = args.embed_dim
    seqlen = args.seqlen
    topk_count = args.topk_count
    cuda = args.cuda
    architecture = [int(h) for h in args.architecture.split(",")]

    nlen = 1 + (n-1) * len(architecture)

    # give model a timestamp; set up artifacts dirs
    model_name = "%08x" % int(time.time())
    setup_model_artifacts(artifactpath, model_name)

    # init tensorbaord writer
    writer = SummaryWriter(log_dir=f"{artifactpath}/{model_name}/tensorboard/")

    # init detections file (used to eyeball sigs while model trains)
    det_file = open(f"{artifactpath}/{model_name}/det_file", "w")

    # init model
    model = StrScorer(architecture=architecture, lr=lr, n=n, embed_dim=embed_dim, topk_count=topk_count, cuda=cuda)

    # init replay buffer
    rpbuffer = RpBuffer(timeout=3600)

    niter = 0
    warmup = 0
    while True:
        # create a new data loader
        dl = torch.utils.data.DataLoader(BinaryDataLoader(uripath, balance=True),
                batch_size=32, shuffle=True, num_workers=8)

        # one-time filling of replay buffer until we have ~3200 binaries to sample from
        if warmup == 0:
            for sample_bytes, shas, labels in dl:
                print("warmup", warmup)
                full_yps, full_labels = update_buffers(model, sample_bytes, shas, labels, seqlen, nlen, rpbuffer, det_file)
                warmup += 1

                if warmup > 100:
                    break

        # grab samples from dataloader
        for sample_bytes, shas, labels in dl:
            # cull replay buffer
            rpbuffer.cull_blocks()

            # get most malicious looking chunks from samples
            full_yps, full_labels = update_buffers(model, sample_bytes, shas, labels, seqlen, nlen, rpbuffer, det_file)
            write_stats(writer, niter, full_yps, full_labels)
            niter += 1
        
            # save model every 1000 iterations
            if niter % 1000 == 0:
                torch.save(model, f"{artifactpath}/{model_name}/model.mdl")

            # sample from replay buffer 40 times (arbitrary number)
            for sample_i in range(40):
                # grab a batch of 32 samples
                samples = rpbuffer.get_samples(32)

                # sample chunks from binaries based on scores of each chunk
                batch_x = []
                batch_y = torch.zeros(32, 1)
                block_sel = []
                sha_samples = []

                for i, s in enumerate(samples):
                    sha_sample = s[1]
                    data = s[2]
                    y = s[3]
                    yps = s[4]

                    # softmax-sampling of chunks.  chunks with higher scores relative
                    # to other chunks are more likely to be selected.
                    yps_softmax = softmax(yps)

                    # select a chunk weighted by softmax probability
                    block_idx = np.random.choice(len(yps), p=yps_softmax)

                    # append chunk, label info, block index, and sha to update score later
                    batch_x.append(data[block_idx])
                    batch_y[i] = y
                    block_sel.append(block_idx)
                    sha_samples.append(sha_sample)

                batch_x = torch.from_numpy(unpackbits(batch_x, seqlen))
                
                # fit model
                yps, loss = model.fit(batch_x, batch_y)
                yps = yps.flatten()

                # update scores in replay buffer with new score of chunk
                for i in range(batch_x.shape[0]):
                    rpbuffer.update_score(sha_samples[i], block_sel[i], yps[i])
