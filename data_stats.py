import base64
import gzip
import json
import numpy as np
from tqdm import tqdm
import logging
import multiprocessing as mp
from os import listdir

logging.basicConfig(filename='stats.log',level=logging.INFO, format='%(asctime)s %(message)s')


def process_train_file(prefix, filename, out):
    n_example = 0
    answerable = 0
    long_ans = 0
    short_ans = 0
    short_no_long = 0
    short_in_long = 0
    short_pos = 0
    short_max = 0
    long_256 = 0
    q_max = 0
    logging.info('[START]'+prefix+filename)
    with open(prefix + filename, "br") as fileobj:
        zip_file = gzip.GzipFile(fileobj=fileobj)
        for line in zip_file:
            n_example += 1
            json_example = json.loads(line)
            q_max = max(q_max, len(json_example["question_tokens"]))
            annotation = json_example["annotations"][0]
            candidate_index = annotation['long_answer']['candidate_index']
            short_answers = annotation['short_answers']
            has_answer = False
            if len(short_answers)>0:
                short_ans += 1
                has_answer = True
                short_answer=short_answers[0]
                short_max = max( short_max, short_answer["end_token"]-short_answer["start_token"])
                if candidate_index != -1:
                    long_answer = json_example["long_answer_candidates"][candidate_index]
                    if long_answer["start_token"]<= short_answer["end_token"] <= long_answer["end_token"]:
                        short_in_long += 1
                        short_pos += 1 if short_answer["end_token"]-long_answer["start_token"]<= 512 else 0
                else:
                    short_no_long += 1
            if candidate_index != -1: # long answer exists
                has_answer = True
                long_ans += 1
                long_answer = json_example["long_answer_candidates"][candidate_index]
                if long_answer["end_token"]-long_answer["start_token"]<=512:
                    long_256+=1
            if has_answer:
                answerable += 1
    logging.info('[FINISH]'+prefix+filename)
    logging.info("{}/{} answerable, {} long answers, {} short answers".format(answerable, n_example, long_ans, short_ans))
    logging.info("Questions {} token max, Short answers {} token max, {} long answer within 512 tokens".format(q_max, short_max, long_256))
    logging.info("{} short questions with no long answers, {} in long answer, pos {}\n".format(short_no_long, short_in_long, short_pos))
    out.put(','.join(map(str, [filename, n_example, answerable, long_ans, short_ans, short_no_long, short_in_long, short_pos, short_max, long_256, q_max])))

def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)

def main():
    prefix = './data/natural_questions/v1.0/train/'
    files = listdir(prefix)
    with open('stats-512.csv', 'w') as csv:
        for i in range(0, len(files), 8):
            output = mp.Queue()
            processes = [mp.Process(target=process_train_file, args=(prefix, f, output)) for f in files[i:i+8]]
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            results = [output.get()+'\n' for p in processes]
            csv.writelines(results)


if __name__ == '__main__':
    main()
