import spacy
from argparse import Namespace
from methods import system_stats, load_json, sentence_stats, index_from_file
from global_recall import most_frequent_omissions, get_count_list, percentiles
from local_recall import local_recall_counts, local_recall_scores
from nouns_pps import pp_stats, compound_stats
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
import argparse

nlp = spacy.load('en_core_web_sm')


def lower_sent(sentence, tagged=False):
    "Make sentence lowercase."
    if tagged:
        return [(word.lower(), pos) for word, pos in sentence]
    else:
        return [word.lower() for word in sentence]


def get_sentences(data, lower=True, tagged=False):
    "Get a list of tokenized sentences from generated output."
    key = 'tagged' if tagged else 'tokenized'
    sentences = [entry[key] for entry in data]
    if lower:
        return [lower_sent(sent, tagged) for sent in sentences]
    else:
        return sentences


def compounds_from_doc(doc):
    "Return a list of compounds from the document."
    compounds = []
    current = []
    for token in doc:
        if token.tag_.startswith('NN'):
            current.append(token.orth_)
        elif len(current) == 1:
            current = []
        elif len(current) > 1:
            compounds.append(current)
            current = []
    if len(current) > 1:
        compounds.append(current)
    return compounds


def annotate_data(source_file, tag=False, compounds=False):
    "Function to annotate existing coco data"
    data = load_json(source_file)
    for entry in data:
        raw_description = entry['caption']
        doc = nlp.tokenizer(raw_description)
        entry['tokenized'] = [tok.orth_ for tok in doc]
        if tag:
            # Call the tagger on the document.
            nlp.tagger(doc)
            entry['tagged'] = [(tok.orth_, tok.tag_) for tok in doc]
        if compounds:
            list_of_compounds = compounds_from_doc(doc)
            entry['compounds'] = list_of_compounds
    return data


def run_all(args):
    "Run all metrics on the data and save JSON files with the results."
    # Annotate generated data.
    annotated = annotate_data(args.source_file,
                              tag=True,
                              compounds=True)

    # Load training data. (For computing novelty.)
    train_data = load_json('./Data/COCO/Processed/tokenized_train2014.json')
    train_descriptions = [
            entry['caption'] for entry in train_data['annotations']
        ]

    # Load annotated data.
    sentences = get_sentences(annotated, lower=True, tagged=False)

    # Analyze the data.
    stats = system_stats(sentences)

    # Get raw descriptions.
    gen_descriptions = [
            entry['caption'] for entry in load_json(args.source_file)
        ]
    extra_stats = sentence_stats(train_descriptions, gen_descriptions)
    stats.update(extra_stats)

    # Save statistics data.

    ################################
    # Global recall

    train_stats = load_json('./Data/COCO/Processed/train_stats.json')
    val_stats = load_json('./Data/COCO/Processed/val_stats.json')

    train     = set(train_stats['types'])
    val       = set(val_stats['types'])
    learnable = train & val

    gen = set(stats['types'])
    recalled = gen & val

    coverage = {"recalled": recalled,
                "score": len(recalled)/len(learnable),
                "not_in_val": gen - learnable}

    coverage['omissions'] = most_frequent_omissions(coverage['recalled'],
                                                    val_stats,         # Use validation set as reference.
                                                    n=None)
    val_count_list = get_count_list(val_stats)
    coverage['percentiles'] = percentiles(val_count_list, recalled)

    ####################################
    # Local recall

    val_index = index_from_file('./Data/COCO/Processed/tagged_val2014.json', tagged=True, lower=True)
    generated = {entry['image_id']: entry['tokenized'] for entry in annotated}
    local_recall_res = dict(scores = local_recall_scores(generated, val_index),
                            counts = local_recall_counts(generated, val_index))

    ##################################
    # Nouns pps
    npdata = {'pp_data': pp_stats(annotated), 'compound_data': compound_stats(annotated)}

    return annotated, stats, coverage, local_recall_res, npdata


def main(args):

    files = glob(os.path.join(args.input_dir, '*.json'))

    t = tqdm(files)
    results = pd.DataFrame()

    for file in t:

        # display the file being processed
        method = os.path.split(file)[-1]
        method = method.replace('_cleaned', '').replace('.json', '')

        t.set_description(method)
        t.refresh()

        annotated, stats, coverage, local_recall_res, npdata = run_all(
            Namespace(source_file=file)
        )

        file_stats = pd.Series({
            'num_types' : stats['num_types'],
            'num_tokens' : stats['num_tokens'],
            'average_sentence_length' : stats['average_sentence_length'],
            'std_sentence_length' : stats['std_sentence_length'],
            'type_token_ratio' : stats['type_token_ratio'],
            'bittr' : stats['bittr'],
            'trittr' : stats['trittr'],
            'percentage_novel' : stats['percentage_novel'],
            'coverage' : coverage['score'],
            'loc5' : local_recall_res['scores'][4]
        }, name=method)

        results = results.append(file_stats)

    results.index = results.index.rename('file')

    results.to_csv(args.out_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        help='the directory of the files to be processed')
    parser.add_argument('--out_file', type=str,
                        default='./diversity_results.csv',
                        help='path for the output csv file')
    args = parser.parse_args()

    main(args)
