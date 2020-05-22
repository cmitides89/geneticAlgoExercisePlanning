import numpy as np
import operator as op
import pandas as pd
from pandas.io.json import json_normalize
import pprint
import random
# import ga_fit_func as gff
import time
# import experimental_mod as exmod
# from tqdm import tqdm as loadbar
# from sklearn import preprocessing
from timeit import default_timer as timer
from ast import literal_eval
# TODO: make ex_gene_pool pull from a database
# TODO: use decorators on these functions that do the same thing.
class Genetic_Class:

    ex_gene_pool = pd.read_csv('cleaned_ex_genes.csv', index_col=0)

    def __init__(self, usr_lvl, goal, no_days, no_exs):
        self.usr_lvl = usr_lvl
        self.goal = goal
        self.no_days = int(no_days)
        self.ex_len = int(no_exs)

# combined total,lower,upper from new_ga
    def create_genes(self, gene_type = None):
        if gene_type:
            ex_genes = self.ex_gene_pool[(self.ex_gene_pool['muscle_group'] == gene_type)&(self.ex_gene_pool['level'] == self.usr_lvl)]
        else:
            ex_genes = self.ex_gene_pool[self.ex_gene_pool['level'] == self.usr_lvl]
        ex_df = ex_genes.sample(n= self.ex_len)

        conditions = [
            ex_df['movement_size'] == 'Largest',
            ex_df['movement_size'] == 'Large',
            ex_df['movement_size'] == 'Medium',
            ex_df['movement_size'] == 'Small'
        ]
        sets_choices = [
            np.random.randint(2, 12, size=1), np.random.randint(
                3, 4, size=1), np.random.randint(3, 5, size=1), np.random.randint(3, 5, size=1)
        ]
        reps_choices = [
            np.random.randint(1, 5, size=1), np.random.randint(
                4, 6, size=1), np.random.randint(6, 11, size=1), np.random.randint(8, 13, size=1)
        ]
        ex_df['sets'] = np.select(conditions, sets_choices, default=0)
        ex_df['reps'] = np.select(conditions, reps_choices, default=0)
        ex_df = ex_df.reset_index(drop=True)

        return ex_df.to_dict('records')


    # def start_evolution():
