from __future__ import division
import pickle
from pprint import pprint
# from config import *
from w2v_dataset_helpers import load_models, print_details, print_latex_version
from collections import OrderedDict

"""
    evaluation script for the analogies task 
    @author: Gerhard Wohlgenannt (2017), ITMO University, St.Petersburg, Russia   

    what happens: a) read evaluation data, 
                  b) for any word embedding collect the number of correct / incorrect results
                  c) print the number of correct / incorrect results 
"""
BOOK_NAME, METHODS, NGRAMS, ANALOGIES_FILE, DOESNT_MATCH_FILE, ANALOGIES_SECTIONS = '', '', '', '', '', ''

# read the analogies evaluation data from the path defined in config.py
# analogies_data = open(ANALOGIES_FILE).readlines()

# evaluate each of the embedding methods defined in config.py
def evaluate_analogies(method, emb_type, result_file):

    # load model and init our data capture variables
    model = load_models(method, emb_type)
    acc_res = model.accuracy(ANALOGIES_FILE)
    results = OrderedDict()

    for res in acc_res:

        ## collect data about this task
        sec = res['section']
        # print("\n\tSection:", sec)
        result_file.write("\n\tSection:" + sec)
        results[sec]={}
        if len(res['correct'])+len(res['incorrect']) != 0:
            results[sec]['perc'] = len(res['correct']) / ( len(res['correct'])+len(res['incorrect']))
        results[sec]['correct']  = len(res['correct'])
        results[sec]['incorrect'] = len(res['incorrect'])
        results[sec]['counts'] = str( len(res['correct'])+len(res['incorrect']) )

    return results


def analogies_evaluation(book_name):
    set_config(book_name.upper())

    # evaluate each of the embedding methods defined in config.py
    file_name = './evaluation_results/' + book_name + '_result_analogies.txt'
    with open(file_name, "w", encoding="utf8") as result_file:
        for (method,emb_type) in METHODS:
            results = evaluate_analogies(method, emb_type, result_file)
            pprint(dict(results), result_file)
            # print("Number of sections:", len(results)-1)
            result_file.write("Number of sections:" + str(len(results)-1) + '\n')
            print_latex_version(results, method, ANALOGIES_SECTIONS, result_file)
            print('done')


def set_config(book_name):
    global METHODS, NGRAMS, ANALOGIES_FILE, ANALOGIES_SECTIONS, DOESNT_MATCH_FILE, DOESNT_MATCH_SECTIONS
    BOOK_SERIES = book_name
    if BOOK_SERIES == "ASOIF":
        METHODS = [
            ## word2vec bash constructed models
            ('asoif_default_w2v', 'bin'),
            ('asoif_w2v_1', 'bin'),
            ('asoif_w2v_2', 'bin'),
            ('asoif_w2v_3', 'bin'),
            ('asoif_w2v_4', 'bin'),

            ## FastText bash constructed models
            ('asoif_default_ft', 'bin'),
            ('asoif_ft_1', 'bin'),
            ('asoif_ft_2', 'bin'),
        ]

        if NGRAMS:
            METHODS = [
                # ('ppmi', 'bin'), #ppmi
                ('asoif_w2v-ww12-300-ngram', 'bin'),
                ## Skip-gram, window-size 12, 300dim, hier.softmax, iter 15, no neg-sampling
                ('asoif_w2v-ww12-300-ns-ngram', 'bin'),
                ## Skip-gram, window-size 12, 300dim, hier.softmax, iter 15, -negative 15
                ('asoif_fastText_ngram', 'vec'),  # default and: -epoch 25 -ws 12
                ('asoif_lexvec_ngram', 'vec'),  # default and: -epoch 25 -ws 12
            ]

    if BOOK_SERIES == "HP":
        METHODS = [
            ## word2vec bash constructed models
            ('hp_default_w2v', 'bin'),
            ('hp_w2v_1', 'bin'),
            # ('hp_w2v_2', 'bin'),
            # ('hp_w2v_3', 'bin'),
            # ('hp_w2v_4', 'bin'),
            #
            # ## FastText bash constructed models
            # ('hp_default_ft', 'bin'),
            # ('hp_ft_1', 'bin'),
            # ('hp_ft_2', 'bin'),
        ]

        if NGRAMS:
            METHODS = [
                # ('ppmi', 'bin'), #ppmi
                ('hp_lexvec_ngram', 'vec'),
                ('hp_fastText_ngram', 'vec'),  # for paper!, 25 epoch
                ('hp_w2v-default-ngram', 'bin'),
                ('hp_w2v-ww12-300-ngram', 'bin'),
                ('hp_w2v-ww12-300-ns-ngram', 'bin'),
                # ('hp_glove_ngrams', 'vec'),
                # ('hp_w2v-CBOW_ngrams', 'bin'),
            ]

    if BOOK_SERIES == "SH":
        METHODS = [
            ('sherlock_holmes', 'vec'),
        ]

        # -----------------------------------------------------
    # for "doesnt_match" evaluation script
    # -----------------------------------------------------

    if BOOK_SERIES == "ASOIF":
        PRINT_DETAILS = False  ## verbose debugging of eval results

        if NGRAMS:
            ANALOGIES_FILE = "./datasets/questions_soiaf_analogies_ngram.txt"
            DOESNT_MATCH_FILE = "./datasets/questions_soiaf_doesnt_match_ngram.txt"
            ANALOGIES_SECTIONS = ['name-nickname', 'child-father', 'total']
            DOESNT_MATCH_SECTIONS = [': bays', ': gods', ': cities-fortresses', ': Maesters', ': Houses', 'TOTAL']
            FREQ_FILE = "./datasets/freq_asoif_ngram.pickle"

        else:
            ANALOGIES_FILE = "./datasets/questions_soiaf_analogies_rus.txt"
            DOESNT_MATCH_FILE = "./datasets/questions_soiaf_doesnt_match_rus.txt"
            ANALOGIES_SECTIONS = ['firstname-lastname', 'child-father', 'husband-wife', 'geo-name-location',
                                  'houses-seats', 'total']
            DOESNT_MATCH_SECTIONS = [': family-siblings', ': names-of-houses', ': Stark clan', ': free cities', 'TOTAL']
            FREQ_FILE = "./datasets/freq_soiaf.pickle"

        ### which sections to show in the paper..

    if BOOK_SERIES == "HP":
        PRINT_DETAILS = False  ## verbose debugging of eval results

        if NGRAMS:
            ANALOGIES_FILE = "./datasets/questions_hp_analogies_ngram.txt"
            DOESNT_MATCH_FILE = "./datasets/questions_hp_doesnt_match_ngram.txt"
            # ANALOGIES_SECTIONS = ['Gryffindor-Quidditch-team', 'Yule_ball-gentleman-lady', 'character-where_they_work', 'character-creature', 'total']
            ANALOGIES_SECTIONS = ['character-creature', 'character-where_they_work', 'total']
            # DOESNT_MATCH_SECTIONS = [': geographical-objects', ': closest-friends', ': unforgivable-curses', ': members-of-Order_of_the_Phoenix', ': ministers-for-magic', 'TOTAL']
            DOESNT_MATCH_SECTIONS = [': geographical-objects', ': ministry_of_magic-employees',
                                     ': members-of-Order_of_the_Phoenix', 'TOTAL']
            FREQ_FILE = "./datasets/freq_hp_ngram.pickle"
        else:
            ANALOGIES_FILE = "./datasets/questions_hp_analogies_rus.txt"
            DOESNT_MATCH_FILE = "./datasets/questions_hp_doesnt_match_rus.txt"
            ANALOGIES_SECTIONS = ['firstname-lastname', 'child-father', 'husband-wife', 'name-species', 'total']
            # DOESNT_MATCH_SECTIONS = [': family-members', ': Gryffindor-members', ': magic-creatures', ': wizards-animagi', 'TOTAL']
            DOESNT_MATCH_SECTIONS = [': family-members', ': Gryffindor-members', ': magic-creatures', ': professors',
                                     'TOTAL']
            FREQ_FILE = "./datasets/freq_hp.pickle"