import gzip

with gzip.open('./data/v1.0_sample_nq-dev-sample.jsonl.gz', 'rb') as f:
    with gzip.open('./data/tiny-dev.jsonl.gz', 'wb') as tof:
        keep = 10
        read = 0
        for line in f:
            if read >= keep:
                break
            tof.write(line)
            read += 1