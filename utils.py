import time
import torch
import random
import boto3
import gzip

import numpy as np

from urllib.parse import urlparse

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# class for getting binaries from s3 or local filesystem as a
# torch dataset generator
class BinaryDataLoader(torch.utils.data.Dataset):
    def __init__(self, uri_path, balance=True):

        self.s3_client = None
        
        # get uris and labels to train on
        uris, labels = [], []
        for l in open(uri_path):
            uri, label = l[:-1].split(",")
            label = float(label)

            uris.append(uri)
            labels.append(label)

        # convert to np arrays for easier indexing
        uris = np.asarray(uris)
        labels = np.asarray(labels)

        # split into benign and malicious uris
        benign_uris = uris[labels == 0]
        malicious_uris = uris[labels == 1]

        # balance the dataset by throwing away samples in the majority class
        if balance:
            if len(benign_uris) > len(malicious_uris):
                benign_idxs = np.random.permutation(len(benign_uris))
                benign_uris = benign_uris[benign_idxs[:len(malicious_uris)]]
            else:
                malicious_idxs = np.random.permutation(len(malicious_uris))
                malicious_uris = malicious_uris[malicious_idxs[:len(malicious_uris)]]

        # finally, stitch everything together
        self.uris = np.concatenate([malicious_uris, benign_uris])
        self.labels = np.concatenate([np.ones(len(malicious_uris)), np.zeros(len(benign_uris))]).astype(np.float32)

        return

    def __len__(self):
        return(len(self.uris))

    def __getitem__(self, index):
        # grab a uri and label
        uri = self.uris[index]
        label = self.labels[index]

        # check if URI looks like an s3 key.  if so, download key from s3.
        if uri[:5] == "s3://":
            # init s3 client if not yet done
            if self.s3_client is None:
                self.s3_client = boto3.Session().client('s3')

            # split s3 uri into bucket and key
            uri_parsed = urlparse(uri, allow_fragments=False)
            bucket, key = uri_parsed.netloc, uri_parsed.path[1:]

            # download sample from s3
            sample_bytes = self.s3_client.get_object(Bucket=bucket, Key=key, RequestPayer='requester')["Body"].read()
        # otherwise, open from filesystem
        else:
            sample_bytes = open(uri, "rb").read()

        # if the sample is gzip'd, decompress it.
        if sample_bytes[:2] == b"\x1f\x8b":
            sample_bytes = gzip.decompress(sample_bytes)

        return sample_bytes, uri, float(label)

# class for the "replay buffer"
#
# computing scores for every chunk in a large binary is expensive.  to speed things up,
# we keep track of a "replay buffer" of entire binaries and scores that we can sample
# from.
#
# "timeout" variable controls how long to keep samples around for (seconds).
class RpBuffer():
    def __init__(self, timeout=600):
        self.blocks = dict()
        self.timeout = timeout

    # delete samples in the replay buffer that are older than "timeout" seconds.
    def cull_blocks(self):
        to_delete = set()

        # create a list of samples that are older than self.timeout
        curr_t = time.time()
        for block in self.blocks:
            block_t, sha, data, label, yps = self.blocks[block]
            if curr_t - block_t > self.timeout:
                to_delete.add(block)

        # print an informative message
        if len(to_delete) > 0:
            print("** Clearing out %d blocks from replay buffer**" % len(to_delete))

        # clear out the blocks
        for block in to_delete:
            del self.blocks[block]

    # add a sample to the replay buffer
    def add_block(self, sha, data, label, yps):
        self.blocks[sha] = (time.time(), sha, data, label, yps)

    # update the score for a block of a sample in the replay buffer
    def update_score(self, sha, block_idx, yp_new):
        block_t, sha, data, label, yps = self.blocks[sha]
        yps[block_idx] = yp_new
        self.blocks[sha] = (block_t, sha, data, label, yps)

    # sample data from the replay buffer
    def get_samples(self, nsamples):
        # random sampling
        samples = [self.blocks[k] for k in self.blocks]
        random.shuffle(samples)
        return samples[:nsamples]

# helper function for turning a sequence of bytes into a sequence of int64's for torch's
# embedding layer
def unpackbits(sample_bytes, seqlen):
    x = np.zeros((len(sample_bytes), seqlen), dtype=np.int64)

    for i in range(len(sample_bytes)):
        b = np.frombuffer(sample_bytes[i], dtype=np.uint8)
        x[i,:len(b)] = b

    # 0 is reserved for padding -- return bytes in the range of 1-257
    return x + 1

# helper function for softmax computation.
def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    x = x / np.sum(x)

    return x

# take a batch of binaries, feed them through the model, and save them off to the replay buffer.
def update_buffers(model, sample_bytes, shas, labels, seqlen, nlen, rpbuffer, det_file):
    full_yps = np.zeros(len(sample_bytes))
    full_labels = np.zeros(len(sample_bytes))

    for idx, (s, sha, l) in enumerate(zip(sample_bytes, shas, labels)):
        # chunk the sample up into seqlen-byte chunks.
        offset_idxs = list(range(int(np.ceil(len(s) / seqlen))))
        seqs = [s[o*seqlen:(o+1)*seqlen] for o in offset_idxs]

        # align last seq with end of file so we don't have a bunch of blocks
        # with empty stuff at the end
        if len(seqs) > 1:
            seqs[-1] = s[-seqlen:]

        # init storage for max score of each chunk
        yps = np.zeros(len(seqs))

        # keep track of max output and sig associated with output
        max_yp = None
        max_sig = b""

        # iterate through the chunks
        for b in range(len(seqs)):
            # if the chunk is smaller than the convolutional receptive field, we won't
            # get any scores.  skip it.
            if len(seqs[b]) < nlen:
                continue
            seq = seqs[b]

            # does this do anything for memory/speed? maybe??
            with torch.no_grad():
                # feed forward the sample
                ngrams = torch.from_numpy(unpackbits([seq], seqlen))
                yp = model.forward(ngrams)
                yp = yp.detach().cpu().numpy().squeeze()[:len(seq)-nlen+1]

            # get the sig and max score for the chunk
            max_sig_idx = np.argmax(yp)
            yp = yp[max_sig_idx]

            # update the max score/sig
            if max_yp is None or yp > max_yp:
                max_yp = yp
                max_sig = seq[max_sig_idx:max_sig_idx+nlen]

            # fill in the max score for the chunk
            yps[b] = yp

        # final score of the sample: max score over all the whole sample.
        full_yps[idx] = np.max(yps)
        full_labels[idx] = l

        # add whole sample to replay buffer
        rpbuffer.add_block(sha, seqs, l, yps)

        # write out information to the detections file
        det_file.write("%d %s %0.3f %s %s\n" % (l, sha, max_yp, max_sig.hex(), str(max_sig)))

    return full_yps, full_labels

# helper function to update tensorboard plots/print stuff to stdout
def write_stats(writer, niter, full_yp, labels):
    # compute accuracy across good and bad
    y = labels.flatten()
    acc_good = np.mean((full_yp[y < 0.5] < 0))
    acc_bad = np.mean((full_yp[y > 0.5] > 0))
    acc = (acc_good + acc_bad) / 2

    print(niter, acc, acc_good, acc_bad)

    writer.add_scalar("acc/yp", acc, niter)
    writer.add_scalar("acc/good" , acc_good, niter)
    writer.add_scalar("acc/bad" , acc_bad, niter)
