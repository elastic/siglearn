import argparse
import glob
import gzip
import torch
import random

import numpy as np

from model import StrScorer
from utils import BinaryDataLoader

def sigm(x):
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate signatures given a trained model and a corpus of samples.")
    parser.add_argument("--use_cuda", help="use cuda (y/n)?")
    parser.add_argument("--verbose", help="print out potential signatures that are above the score threshold (y/n)")
    parser.add_argument("--model_path", help="filename of pretrained model")
    parser.add_argument("--score_threshold", help="threshold for malicious signature score", default=4.0, type=float)
    parser.add_argument("--sample_path", help="path for samples to create rules for")
    parser.add_argument("--yara_filename", help="filename for yara rule")
    args = parser.parse_args()

    model_name = args.model_path
    score_threshold = args.score_threshold
    sample_path = args.sample_path
    yara_filename = args.yara_filename
    use_cuda = args.use_cuda
    verbose = args.verbose

    if verbose is None or verbose == 'n':
        verbose = False
    elif verbose == 'y':
        verbose = True
    
    if use_cuda is None or use_cuda == 'n':
        use_cuda = False
    elif use_cuda == 'y':
        use_cuda = True

    print("loading model %s..." % model_name)

    print("loading model %s..." % model_name)
    if use_cuda:
        model = torch.load(model_name, map_location=torch.device('cuda'))
    else:
        model = torch.load(model_name, map_location=torch.device('cpu'))
        model.use_cuda = False
    print("done")

    hidden = len(model.local_layers) // 2
    n = model.local_layers[0].kernel_size[0]
    nlen = 1 + (n-1) * hidden

    uris = glob.glob(sample_path + "/*")

    success = 0

    print("attempting to extract sigs for %d samples..." % len(uris))

    sigs = []

    for uri in uris:
        # open sample
        sample = open(uri, "rb").read()

        best_yp = None
        best_sig = b""
        best_offset = None

        model.eval()
        # get sigs 1mb at a time
        for i in range(0, len(sample), 1000000):
            subsample = sample[i:i+1000000]

            # skip if the subsample is too short
            if len(subsample) < nlen:
                continue

            ss = torch.from_numpy(np.frombuffer(subsample, dtype=np.uint8).reshape(1,-1).astype(np.int64)) + 1

            if use_cuda:
                ss = ss.cuda()

            yp = model.forward(ss).detach().cpu().numpy().flatten()

            # print out potential sigs
            if verbose:
                potential_idxs = np.where(yp > score_threshold)[0]
                for idx in potential_idxs:
                    print("potential signature: (score=%0.3f, offset=%08x) {%s}" % (yp[idx], idx+i, subsample[idx:idx+nlen]))

            if best_yp is None or yp.max() > best_yp:
                idx = yp.argmax()
                best_yp = yp.max()
                best_sig = subsample[idx:idx+nlen]
                best_offset = idx + i

        if best_sig is not None and best_yp > score_threshold:
            sigs.append(best_sig)

        print("%s best sig (score=%0.3f, offset=%08x): {%s}" % (uri, best_yp, best_offset, best_sig))

    print("signature success rate: (%d/%d)" % (len(sigs), len(uris)))
    print("writing out yara rule to %s" % yara_filename)

    # dump out yara rule
    with open(yara_filename, "w") as fid:
        for sig in sigs:
            fid.write("rule AUTO_%s: {\n" % sig.hex().replace('?','q'))
            fid.write("    strings:\n")
            fid.write("        $a = {%s}\n" % sig.hex())
            fid.write("    condition:\n")
            fid.write("        all of them\n")
            fid.write("}\n\n")
