from __future__ import division, print_function
import pickle, random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from collections import OrderedDict
from pprint import pprint
# from config import *
from w2v_dataset_helpers import load_models, print_details, print_latex_version
#from SVD_doesntmatch import *
#from ppmi_doesnt_match import *

SEP_I = 4 ## position of the seperator symbol in the input data

"""
    evaluation script for the doesnt_match() task 
    @author: Gerhard Wohlgenannt (2017), ITMO University, St.Petersburg, Russia   

    what happens: a) read evaluation data, 
                  b) for any word embedding collect the number of correct / incorrect results
                  c) Collect and analyse results with pandas dataframes
"""
BOOK_NAME, METHODS, NGRAMS, ANALOGIES_FILE, DOESNT_MATCH_FILE, ANALOGIES_SECTIONS = '', '', '', '', '', ''
# read the doesnt_match evaluation data from the path defined in config.py

# doesnt_match_data = open(DOESNT_MATCH_FILE).readlines()

def evaluate_doesnt_match(method, emb_type, term_freq=None):
    """ create task_results which contains the judgements of all input questions (task units) """

    task_results = []
   
    if method == 'ppmi':
        ## currently not included into the public version   
        ## just use standard ppmi, or ask github owner for sending the file        
        ppmi = create_matrix()
    
    else:
        # load model and init our data capture variables
        model = load_models(method, emb_type)
     
    for line in doesnt_match_data:

        # those are the section (or :end) markers
        if line.startswith(":"): 
            task_type = line.strip()  
            continue
        
        ### get information from the input line
        ### input line format is: task-terms :: outlier
        line_list = line.strip().split()

        assert(len(line_list)) == 6

        ## just split up and a assign the input data
        task_terms, difficulty, correct_outlier = line_list[:SEP_I], line_list[SEP_I][2], line_list[SEP_I+1:][0]

        ## call gensim model to find the outlier candidate
        if method == 'ppmi':
            found_outlier = solve_task( ppmi, task_terms) #, on_error='skip') 
            if not found_outlier: continue
            #found_outlier = solve_task_2( ppmi, task_terms) 
        else:
            print(task_terms)
            found_outlier = model.doesnt_match( task_terms )


        ## judge correctness of candidate 
        correct=0.0 # False
        if found_outlier == correct_outlier:
            correct=1.0 # True

        if DO_FREQ_EVAL:
            ## compute avg term frequency of the task_terms
            try:
                list(term_freq[t] for t in task_terms)
            except KeyError as e:
                print("""
        Looks like you modified the evaluation questions.
        You either need to update freq_ files in datasets directory,
        or set DO_FREQ_EVAL=False.
                    """)
                sys.exit()

            avg_tf = sum(term_freq[t] for t in task_terms) / len(task_terms)
            #print(list(term_freq[t] for t in task_terms))
            #print (avg_tf)

            task_results.append( (task_type, task_terms, found_outlier, correct_outlier, term_freq[found_outlier], term_freq[correct_outlier], avg_tf, difficulty, correct) )
        else:
            task_results.append( (task_type, task_terms, found_outlier, correct_outlier, difficulty, correct) )

        # if DO_FREQ_EVAL and not correct:
        #     print('XXX: {:50} {:10} {:10} {:5} {:5} {:8} {:5} {:5}'.format(task_terms, found_outlier, correct_outlier, term_freq[found_outlier], term_freq[correct_outlier], avg_tf, difficulty, correct))


    return task_results


def analyze_with_pandas(method, task_results, result_file):

        df = pd.DataFrame(task_results)
        if DO_FREQ_EVAL:
            df.columns = ['task_type', 'task_terms', 'found_outlier', 'correct_outlier', 'found_tf', 'correct_tf', 'avg_tf', 'difficulty', 'correct']
        else:
            df.columns = ['task_type', 'task_terms', 'found_outlier', 'correct_outlier', 'difficulty', 'correct']

        ### 1.) collect the generate percentage of correct answers per section and in total
        results = OrderedDict()

        ## in the analysis of percentages we don't want the frequency information to confuse the results, so we remove frequency data :) 
        tmp_df = df[['task_type', 'task_terms', 'found_outlier', 'correct_outlier', 'difficulty', 'correct']].copy()
        gb_tt = tmp_df.groupby('task_type')
        #print (gb.mean(), gb_tt.count())
        for name, group in gb_tt:
            #print ("YYYYY", group.mean())
            results[name] = {'counts': group['task_terms'].count(), 'perc': group.mean()[0]}

        results['TOTAL'] = {'counts': df['correct'].count(), 'perc': df['correct'].mean() }

 
        #for section in set(df['task_type'].tolist()):
        #     print(section)
        #     a = df[df.task_type == section]
        #     print  a[a.correct == True].shape[0]
        #     print  a[a.correct == False].shape[0]


        print("\nTotal values of accuracy per **difficulty** category for: " + method )
        result_file.write('\n')
        result_file.write("\nTotal values of accuracy per **difficulty** category for: " + method)
        gb_diff = df.groupby('difficulty')
        print(gb_diff.mean())
        result_file.write('\n' + str(gb_diff.mean()))


        ## analyze data from the most difficult class // per section
        print("\nDeeper look into difficulty data, here only check out data for difficulty level: 1")
        result_file.write("\nDeeper look into difficulty data, here only check out data for difficulty level: 1")
        tt = gb_diff.get_group('1').groupby('task_type')
        print (tt['correct'].count())
        result_file.write(str(tt['correct'].count()))
        print (tt.mean())
        result_file.write(str(tt.mean()))
        tt.mean()

        ##### deeper overview
        print("\nDeeper look into difficulty data, here check out data for all difficulty levels")
        result_file.write("\nDeeper look into difficulty data, here check out data for all difficulty levels")
        deep = df.groupby( ['task_type', 'difficulty'] )
        print(deep.mean())
        result_file.write(str(deep.mean()))

        print("\ndf.describe -- closer statistical look at global data")
        result_file.write("\ndf.describe -- closer statistical look at global data")
        print(df.describe())
        result_file.write(str(df.describe()))

        print('Number of categories',  len(gb_tt))
        result_file.write('Number of categories' +  str(len(gb_tt)))

        if DO_FREQ_EVAL:
            # ------------------------------------------- NEW ---------------------------------------------------------------------------#
            ## bin frequency into brackets
            bins = [0, 20, 50, 100, 500, 1000, 1000000]
            group_names = ['1', '2', '3', '4', '5', '6']

            df['found_tf_category'] = pd.cut(df['found_tf'], bins, labels=group_names)
            df['avg_tf_category'] = pd.cut(df['avg_tf'], bins, labels=group_names)
            df['correct_tf_category'] = pd.cut(df['correct_tf'], bins, labels=group_names)
            #print(df.head())

            ## correlations 
            print("correlation between difficulty and correct result", df['difficulty'].astype(int).corr(df['correct']))
            result_file.write("correlation between difficulty and correct result" + str(df['difficulty'].astype(int).corr(df['correct'])))
            print("correlation between correct-term frequency and correctness", df['correct_tf'].corr(df['correct']))
            result_file.write("correlation between   found-term frequency and correctness" +  str(df['found_tf'].corr(df['correct'])))
            print("correlation between average term frequency and correctness", df['avg_tf'].corr(df['correct']))
            result_file.write("correlation between average term frequency and correctness" + str(df['avg_tf'].corr(df['correct'])))
            print("correlation between frequency bin and correctness", df['found_tf_category'].astype(int).corr(df['correct']))
            result_file.write("correlation between frequency bin and correctness" + str(df['found_tf_category'].astype(int).corr(df['correct'])))
            print("correlation between avg_frequency bin and correctness", df['avg_tf_category'].astype(int).corr(df['correct']))
            result_file.write("correlation between avg_frequency bin and correctness" + str(df['avg_tf_category'].astype(int).corr(df['correct'])))

            print("\n***************************************************Total values of accuracy per **tf_category** for: " + method)
            result_file.write("\n***************************************************Total values of accuracy per **tf_category** for: " + str(method))

            print('\n------------ found_tf_category')
            result_file.write('\n------------ found_tf_category')
            gb_tf_c = df.groupby('found_tf_category')
            print (gb_tf_c.mean())
            result_file.write(str(gb_tf_c.mean()))
            print (gb_tf_c.size())
            result_file.write(str(gb_tf_c.size()))

            print('\n------------ avg_tf_category')
            result_file.write('\n------------ avg_tf_category')
            gb_avg_tf_c = df.groupby('avg_tf_category')
            print (gb_avg_tf_c.mean())
            result_file.write(str(gb_avg_tf_c.mean()))
            print (gb_avg_tf_c.size())
            result_file.write(str(gb_avg_tf_c.size()))

            print('\n------------ correct_tf_category')
            result_file.write('\n------------ correct_tf_category')
            a = df.groupby('correct_tf_category')
            print (a.mean())
            result_file.write(str(a.mean()))
            print (a.size())
            result_file.write(str(a.size()))

            print("\nTotal values of accuracy per **found_tf_category** category for: " + method)        
            result_file.write("\nTotal values of accuracy per **found_tf_category** category for: " + method)
            gb_diff = df.groupby('found_tf_category')
            print (gb_diff.mean())
            result_file.write(str(gb_diff.mean()))

            print("\nTotal values of accuracy per **avg_tf_category** category for: " + method)        
            result_file.write("\nTotal values of accuracy per **avg_tf_category** category for: " + method)
            gb_diff = df.groupby('avg_tf_category')
            print (gb_diff.mean())
            result_file.write(str(gb_diff.mean()))

            print("\nTotal values of accuracy per **correct_tf_category** category for: " + method)        
            result_file.write("\nTotal values of accuracy per **correct_tf_category** category for: " + method)
            gb_diff = df.groupby('correct_tf_category')
            print (gb_diff.mean())
            result_file.write(str(gb_diff.mean()))

        return results

    
def doesnt_match_evaluation(book_name, freq_eval):
    set_config(book_name.upper())
    global doesnt_match_data
    global DO_FREQ_EVAL
    doesnt_match_data = open(DOESNT_MATCH_FILE).readlines()
    DO_FREQ_EVAL = freq_eval
    if DO_FREQ_EVAL:
        term_freq = pickle.load(open(FREQ_FILE, 'rb'))

    file_name = './evaluation_results/' + book_name + '_result_doesnt_match.txt'
    with open(file_name, "w", encoding="utf8") as result_file:
        result_file.write(str(term_freq))
        print(term_freq)

    # evaluate each of the embedding methods defined in config.py
        for (method,emb_type) in METHODS:
            task_results = evaluate_doesnt_match(method, emb_type, term_freq)
            results = analyze_with_pandas(method, task_results, result_file)

            pprint(dict(results))
            #sys.exit()
            #print(results)
            print_latex_version(results, method, DOESNT_MATCH_SECTIONS, result_file)

            df =  pd.DataFrame(task_results)

def set_config(book_name):
    global METHODS, NGRAMS, ANALOGIES_FILE, ANALOGIES_SECTIONS, DOESNT_MATCH_FILE, DOESNT_MATCH_SECTIONS, DOESNT_MATCH_FILE, FREQ_FILE
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
            ('hp_w2v_2', 'bin'),
            ('hp_w2v_3', 'bin'),
            ('hp_w2v_4', 'bin'),

            ## FastText bash constructed models
            ('hp_default_ft', 'bin'),
            ('hp_ft_1', 'bin'),
            ('hp_ft_2', 'bin'),
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
            ANALOGIES_FILE = "./datasets/questions_soiaf_analogies.txt"
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
