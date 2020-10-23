import numpy as np
import operator as op
import pandas as pd
from pandas.io.json import json_normalize
import pprint
import random
pp = pprint.PrettyPrinter(indent=4)
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
    n_day_cols = ['day', 'day_rating', 'day_type', 'exercises',
                  'ex_l_len', 'goal', 'usr_lvl', 'normalized_score', 'pop_num']
    micro_cols = ['micro_rating', 'workingdays', 'usr_lvl', 'micro_type']

    def __init__(self, usr_lvl, goal, no_days, no_exs):
        self.usr_lvl = usr_lvl
        self.goal = goal
        self.no_days = int(no_days)
        self.ex_len = int(no_exs)
        self.upper_mating_pool = None
        self.lower_mating_pool = None
        self.full_mating_pool = None
        self.micro_mating_pool = None
        self.microcyc_gene_pool = None
        # COMPLETED: set mutation rate
        self.mutation_rate = 0.01
# combined total,lower,upper from new_ga
    def create_genes(self, gene_type = None):
        '''
        gene_type default to None, creates 
        genes for fullbody 
        '''
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

# combined the up day lwday and fb day from new_ga
    def create_day_pheno(self, gene_type):
        '''
        # generate exercises based on gene type aka genes for a day pheno
        '''
        if gene_type == 'lowerbody' or gene_type == 'upperbody':
            exercises = self.create_genes(gene_type)
        else:
            # if the gene type is full body then create genes default gene_type val of None
            exercises = self.create_genes()
        
        # create the day pheno using exercises and gene type as the day type
        day = {'day': '', 'day_rating': 0, 'day_type': gene_type,
            'exercises': exercises, 'ex_l_len': self.ex_len, 'goal': self.goal, 'usr_lvl': self.usr_lvl}

        day_series = pd.Series(day)
        # print(day_series)

        return day_series

# TODO: find a way to merge function for day and microcycle
    def generate_mating_pool(self, day_dna_df):
        '''
        Takes days as dna type: dataframe
        Normalizes score, and creates a population number column 
        returns a dataframe of days with pop nums
        '''
        day_dna_df['day_rating'] = day_dna_df['day_rating'].astype(float)
        # day_dna_df['pop_num'] = day_dna_df['normalized_score'].apply(lambda x : x*100 if x > 0 else 1)
        # DETERMINE MATING PROBABILITY
        conditions = [
            (day_dna_df['day_rating'].values > 0),
            (day_dna_df['day_rating'].values <= 0)
        ]
        
        pos_pop_size = day_dna_df['normalized_score'] * 100
        
        choices = [pos_pop_size, 1]
        
        day_dna_df['pop_num'] = np.select(conditions, choices, default=1)
        day_dna_df = day_dna_df.iloc[day_dna_df.index.repeat(
            day_dna_df['pop_num'].values.astype(int)), :]

        return day_dna_df
    
    def generate_m_mating_pool(self, micro_dna):
        ''' 
        Takes DNA for a Microcycle type: DataFrame 
        Normalizes score, and creates a population number column
        returns dataframe of microcycles with pop nums
        '''
        micro_dna['micro_rating'] = micro_dna['micro_rating'].astype(float)
        # micro_dna['pop_num'] = micro_dna['normalized_score'].apply(lambda x : x*100 if x > 0 else 1)
        # DETERMINE MATING PROBABILITY
        conditions = [
            (micro_dna['micro_rating'].values > 0),
            (micro_dna['micro_rating'].values <= 0)
        ]

        pos_pop_size = micro_dna['normalized_score'] * 100

        choices = [pos_pop_size, 0]

        micro_dna['pop_num'] = np.select(conditions, choices, default=1)
        micro_dna = micro_dna.iloc[micro_dna.index.repeat(
            micro_dna['pop_num'].values.astype(int)), :]
        # micro_dna.to_csv('MICRO_DNA.csv')
        # print('CREATED MICRO POOL')
        return micro_dna    
    
    def create_next_gen(self, day_series):
        '''
        Takes a Day Type: series and based on day_type determines which 
        Mating pool to sample parents from Calls the crossover function and determines mutation using 
        mutation_chance() function
        '''
        if day_series['day_type'] == 'upperbody':
            parents_df = self.upper_mating_pool.sample(n=2)
        elif day_series['day_type'] == 'lowerbody':
            parents_df = self.lower_mating_pool.sample(n=2)
        else:
            # parents_df = mix_mating_pool.sample(n=2)
            print('TODO: add mixed mating pool')
        child = self.day_crossover(parents_df, day_series['ex_l_len'], day_series['usr_lvl'], day_series['goal'])
        child = self.day_mutation_chance(self.mutation_rate, child)
        return child
    

    def day_crossover(self, parent_df, ex_len, usr_lvl, goal):
        '''
        Takes a dataframe of two parents and swaps their dna
        making new child
        '''
        ex_arr1 = np.array(parent_df.iloc[0,:]['exercises'])
        ex_arr2 = np.array(parent_df.iloc[1,:]['exercises'])

        child = parent_df.iloc[0,:].copy()
        rand_midpoint = np.random.randint(low=0,high=ex_len)

        ex_1 = ex_arr1[0:rand_midpoint]
        ex_2 = ex_arr2[rand_midpoint:len(ex_arr2)]

        inhereted_ex = np.concatenate((ex_1,ex_2))
        inhereted_ex = inhereted_ex.tolist()

        child.exercises = inhereted_ex
        # print(type(child['exercises']))
        return child

    def day_mutation_chance(self, mutation_rate, child):
        child_ex_df = pd.DataFrame(child['exercises'])
        child_ex_df['rand_chance'] = np.random.uniform(low=0,high=1,size=child_ex_df.shape[0])
        is_mutatable = child_ex_df['rand_chance'] < mutation_rate
        mutatable_children = child_ex_df[is_mutatable].copy()
        if not mutatable_children.empty:
            # print('BEFORE COMBINE FIRST IN MUTATION:::::')
            # print(child_ex_df)
            mutated_children = self.mutation(mutatable_children)
            # print('MUTATED CHILDREN ::::')
            # print(mutated_children)
            child_ex_df = mutated_children.combine_first(child_ex_df)
            # print('AFTER COMBINE FIRST IN MUTATION::::')
            # print(child_ex_df)
            # time.sleep(7)
            child['exercises'] = child_ex_df.to_dict('records')
        return child

    def mutation(self, mutatable_df):

        mutated_df = pd.DataFrame(columns = mutatable_df.columns)
        mutated_df = mutated_df.append(self.ex_gene_pool.sample(n=len(mutatable_df)), ignore_index=True, sort=True)
        mutated_df.index = mutatable_df.index

        return mutated_df

    def validate_ex_cascade(self, ex_data, goal, usr_lvl):
        mvmnt_arr = ex_data['movement_size'].values
        
        if goal == 'strength' and usr_lvl == 'beginner':
            condition1 = np.array(['Largest', 'Large', 'Medium', 'Medium'])
            rmng_ex = (len(mvmnt_arr) - 4)
            sm = np.array(['Small'])
            condition1 = np.append(condition1, np.repeat(sm, rmng_ex))

            condition2 = np.array(['Largest', 'Large', 'Medium', 'Medium', 'Medium'])
            rmng_ex2 = (len(mvmnt_arr) - 5)
            condition2 = np.append(condition2, np.repeat(sm, rmng_ex2))

            conditions = [(mvmnt_arr == condition1), (mvmnt_arr == condition2)]
            choices = [500, 300]
            result = np.select(conditions, choices, default=-700)

        return result.sum()

    def dup_check(self, ex_data):
        points = 0
        ex_data = pd.value_counts(ex_data['ex_name'].values, sort=False)
        ex_np_arr = np.array(ex_data)
        np.multiply(-10, ex_np_arr, out=ex_np_arr, where = ex_np_arr > 1)
        np.multiply(10, ex_np_arr, out=ex_np_arr, where = ex_np_arr == 1)
        points = ex_np_arr.sum()
        return points
    
    def unqiue_type_check(self, ex_data):
        # TODO:TEST THIS
        # NOTE: NOT CURRENTLY USING, UNSURE IF NEEDED
        equipment = ex_data['ex_equipment']
        ex_data['type_equip'] = ex_data["ex_type"].str.cat(equipment, sep=", ")
        ex_data = pd.value_counts(ex_data['type_equip'].values, sort=False)
        ex_value_arr = np.array(ex_data)
        np.multiply(-10, ex_value_arr, out=ex_value_arr, where = ex_value_arr > 1)
        np.multiply(10, ex_value_arr, out=ex_value_arr, where = ex_value_arr == 1)
        return ex_value_arr.sum()


    def aggregated_day_rating(self, day_series):
        ex_df = pd.DataFrame(day_series['exercises'])
        total_points = 0
        total_points += self.dup_check(ex_df)
        total_points += self.validate_ex_cascade(ex_df,
                                            day_series['goal'], day_series['usr_lvl'])
        # COMPLETED: reimplement fullbody, original code is using lists make it use np array
        if day_series.day_type == 'fullbody':
            # COMPLETED: reimplement this from old code
            total_points += self.rate_fbody_day(ex_df)
        return total_points


    def normalize_day_rating(self, days_df):
        fit_range = np.arange(-3550,2550)
        days_df['normalized_score'] = (days_df['day_rating'] - np.min(fit_range))/(np.max(fit_range) - np.min(fit_range))


    def normalize_micro_rating(self, micro_df):
        fit_range = np.arange(-30200, 17400)
        micro_df['normalized_score'] = (micro_df['micro_rating'] - np.min(fit_range))/(np.max(fit_range) - np.min(fit_range))
    
    
    def create_micro_pheno(self, no_days, usr_lvl, goal):
        working_days = self.microcyc_gene_pool.sample(n=no_days)
        # NOTE: added this conversion DATE: 10/31/2019
        working_days = working_days.to_dict('records')
        microcycle = {'micro_rating':'', 'workingdays': working_days, 'usr_lvl':usr_lvl, 'micro_type':goal}
        return pd.Series(microcycle)

    def assess_4_day(self, wdays):
        """
        Extract the values of the day types
        for every day that aligns with conditions +10 else - 20
        """
        # wdays = pd.DataFrame(micro_series['workingdays'])
        dt_arr = wdays['day_type'].values
        # print(dt_arr)
        condition1 = np.array([
            'upperbody', 'lowerbody', 'upperbody', 'lowerbody'
        ])
        condition2 = np.array([
            'lowerbody', 'upperbody', 'lowerbody', 'upperbody'
        ])
        conditions = [np.array_equal(dt_arr, condition1),
                        np.array_equal(dt_arr, condition2)]

        # choices = [100,100]
        # result = np.select(conditions, choices, default= -400)
        choices = [800, 800]
        result = np.select(conditions, choices, default=-2000)
        
        return result.sum()


    def assess_3_day(self, wdays):
        """
        Extract the values of the day types
        for every day that aligns with conditions +800 else -2000
        """
        dt_arr = wdays['day_type'].values
        condition1 = np.array(['upperbody','fullbody','lowerbody'])
        condition2 = np.array(['lowerbody','fullbody','upperbody'])
        conditions = [np.array_equal(dt_arr, condition1),
                        np.array_equal(dt_arr, condition2)]
        choices = [800,800]
        result = np.select(conditions, choices, default=- 2000)
        return result.sum()

# TODO: implement 2day assessment
    def assess_2_day(self, wdays):
        pass

    def rate_fbody_day(self, ex_df):
        points = 0
        ex_mg = ex_df['muscle_group'].values
        con1 = np.empty((len(ex_df),))
        con2 = np.empty((len(ex_df),))
        con1[::2] ='lowerbody'
        con1[1::2] ='upperbody'
        con2[::2] ='upperbody'
        con2[1::2] ='lowerbody'
        conditions = [
            np.array_equal(ex_mg, con1), np.array_equal(ex_mg, con2)
        ]
        choices = [2500,2500]
        result = np.select(conditions, choices, default=-3500)
        return result.sum()

    def no_dup_days(self, wdays):
        """
        For every day in the working days df
        combine all days exs into one array
        then count the number of times each ex in ex array appears
        they should all be unique.
        for ex len 5 = 50 and so on else -50
        """
        ex_arr = np.array([])
        for index,row in wdays.iterrows():
            ex_df = pd.DataFrame(row['exercises'], columns = ['ex_name',
            'muscle_group','ex_mech_type','ex_type','ex_equipment','level',
            'reps','sets','main-muscle-worked','movement_size','load_size'])
            ex_arr = np.append(ex_arr,ex_df['ex_name'].values)
        val_c = pd.value_counts(ex_arr, sort=False)
        arr_val_c = np.array(val_c)
        np.multiply(-400, arr_val_c, out=arr_val_c, where=arr_val_c > 1)
        np.multiply(200, arr_val_c, out=arr_val_c, where=arr_val_c == 1)
        # print('no dup days sum for 4 days ',arr_val_c.sum())
        return arr_val_c.sum()
    
    def no_dup_days_comparable(self, day, day_pool):
        '''
        Given a day pool, get only the top rated days
        Return a day from day_pool if it's exercises are not the same as day's exercises
        '''
        ex_cols = ['ex_name',
            'muscle_group','ex_mech_type','ex_type','ex_equipment','level',
            'reps','sets','main-muscle-worked','movement_size','load_size']

        # NOTE: may need to change from max to a range
        # make sure the days selected are a good rating
        day_pool = day_pool[day_pool['day_rating'] == day_pool['day_rating'].max()]
        print('===================')
    
    # TODO: try catch block for if return never reached
        for _, day_comp in day_pool.iterrows():
            if not self.compare_day_exs(day, day_comp):
                print('day_comp is the not same as day')
                return day_comp
            
    
    def compare_day_exs(self, day, comp_day):
        day_ex = pd.DataFrame(day['exercises'])
        comp_day_ex = pd.DataFrame(comp_day['exercises'])
        print(day_ex)
        print(comp_day_ex)

        return np.array_equal(day_ex.values, comp_day_ex.values)

    def aggregated_micro_rating(self, micro_series):
        # print(micro_series['workingdays'].columns)
        if isinstance(micro_series['workingdays'], str):
            working_days = literal_eval(micro_series['workingdays'])
            print('converted to list')
            working_days = pd.DataFrame(working_days, columns = self.n_day_cols)
            print('now dataframe')
        else:
            working_days = pd.DataFrame(micro_series['workingdays'], columns = self.n_day_cols)
        total_points = 0
        if working_days.day_rating.values.sum() < 9000:
            total_points += -3000
        else:
            # NOTE ADDED ADDITIONAL POINTS
            total_points += 3500
        total_points += working_days.day_rating.values.sum()
        total_points += self.no_dup_days(working_days)
        if self.no_days == 4:
            total_points += self.assess_4_day(working_days)
        # print(total_points)
        return total_points

    def create_next_gen_micro(self, micro_series):
        # TODO: finish this method add hyp and lean mass
        # if micro_series['micro_type'] == 'Strength':
        parents_df = self.micro_mating_pool.sample(n=2)
        
        child = self.micro_crossover(parents_df, self.usr_lvl, self.goal, self.no_days)
        child = self.micro_mutation_chance(self.mutation_rate, child)
        
        return child

    def micro_crossover(self, parent_df, usr_lvl, goal, no_days):
        day_arr1 = np.array(parent_df.iloc[0,:]['workingdays'])
        day_arr2 = np.array(parent_df.iloc[1,:]['workingdays'])
        child = parent_df.iloc[0,:].copy()
        
        rand_midpoint = np.random.randint(low=0,high=no_days)
        day_1 = day_arr1[0:rand_midpoint]
        day_2 = day_arr2[rand_midpoint:len(day_arr2)]
        inhereted_day = np.concatenate((day_1,day_2))
        
        inhereted_day = inhereted_day.tolist()
        child.workingdays = inhereted_day
        
        return child

    def micro_mutation_chance(self, mutation_rate, child):
        child_micro_df = pd.DataFrame(child['workingdays'], columns = ['day','day_rating','day_type','exercises','ex_l_len','goal','usr_lvl','normalized_score','pop_num'])
        child_micro_df['rand_chance'] = np.random.uniform(low=0,high=1, size=child_micro_df.shape[0])
        is_mutatable = child_micro_df['rand_chance'] < mutation_rate
        
        mutatable_children = child_micro_df[is_mutatable].copy()
        
        if not mutatable_children.empty:
            mutated_children = self.mic_mutation(mutatable_children)
            child_micro_df = mutated_children.combine_first(child_micro_df)
            
            child['workingdays'] = child_micro_df.to_dict('records')
        
        return child

    def mic_mutation(self, mutatable_df):

        mutated_df = pd.DataFrame(columns = mutatable_df.columns)
        mutated_df = mutated_df.append(self.microcyc_gene_pool.sample(n=len(mutatable_df)), ignore_index=True, sort=True)
        mutated_df.index = mutatable_df.index
        return mutated_df

    def micro_3_evolution(self, m_pop_size, dna_days_upper, dna_days_lower):
        pass
    # USED IN START EVO
    def micro_4_evolution(self, m_pop_size, dna_days_upper, dna_days_lower):
        dna_microcycles = pd.DataFrame(index=range(m_pop_size), columns = self.micro_cols)
        self.microcyc_gene_pool = dna_days_upper.append(
            dna_days_lower, ignore_index=True)
        # only choose days that are moderate to high score.
        self.microcyc_gene_pool = self.microcyc_gene_pool[self.microcyc_gene_pool['day_rating'] > 2000]
        dna_microcycles = dna_microcycles.apply(
            lambda x: self.create_micro_pheno(self.no_days, self.usr_lvl, self.goal), axis=1)
        dna_microcycles['micro_rating'] = dna_microcycles.apply(
            self.aggregated_micro_rating, axis=1)

        self.normalize_micro_rating(dna_microcycles)
        self.micro_mating_pool = self.generate_m_mating_pool(dna_microcycles)
        # print(self.micro_mating_pool)
        for _ in range(m_pop_size):
            dna_microcycles = dna_microcycles.apply(self.create_next_gen_micro, axis=1)
            dna_microcycles['micro_rating'] = dna_microcycles.apply(self.aggregated_micro_rating, axis=1)
            self.normalize_micro_rating(dna_microcycles)

            self.micro_mating_pool = self.generate_m_mating_pool(dna_microcycles)
        return dna_microcycles.sort_values(by='micro_rating', ascending=False)
    
    def create_micro(self):
        microcyc = pd.DataFrame(columns=['day','day_rating','day_type','exercises',
        'ex_l_len','goal','usr_lvl','normalized_score','pop_num'])

        '''
        if no_day == 4
            get 1 A_day of day_type y and high rating
            get 1 B_day of day_type y that is not A_day and high rating
            get 1 A_day of day_type z and high rating
            get 1 B_day of day_type z that is not A_day and high rating
        '''
        if self.no_days == 4:
            microcyc.append(self.upper_mating_pool[self.upper_mating_pool['day_rating'] > 2000])

    def start_evolution(self):
        '''
        NOTE DAY DNA GETS INITALIZED AND ASSIGNED HERE [UPPER, LOWER, FULL]
        NOTE POTENTIAL ISSUES: be mindful of the column names, I made them constant vars of the class
        they might cause issues if they are the wrong amount of columns
        original implemntation first created day cols w.o norm score and popnum then eventually added them
        '''

        pop_size = 1000
        m_pop_size = 6
        dna_days_lower = pd.DataFrame(index=range(pop_size), columns=self.n_day_cols)
        dna_days_upper = pd.DataFrame(index=range(pop_size), columns=self.n_day_cols)
        # dna_microcycles = pd.DataFrame(index=range(m_pop_size), columns=self.micro_cols)
        # self.microcyc_gene_pool

        dna_days_upper = dna_days_upper.apply(lambda x: self.create_day_pheno(gene_type = 'upperbody'), axis=1)
        dna_days_lower = dna_days_lower.apply(lambda x: self.create_day_pheno(gene_type = 'lowerbody'), axis=1)

        dna_days_upper['day_rating'] = dna_days_upper.apply(self.aggregated_day_rating, axis=1)
        dna_days_lower['day_rating'] = dna_days_lower.apply(self.aggregated_day_rating, axis=1)

        self.normalize_day_rating(dna_days_upper)
        self.normalize_day_rating(dna_days_lower)

        self.upper_mating_pool = self.generate_mating_pool(dna_days_upper)
        self.lower_mating_pool = self.generate_mating_pool(dna_days_lower)

        score_diff = 0
        score_diff_lw = 0

        prev_score = np.mean(dna_days_upper['normalized_score'])
        prev_score_lw = np.mean(dna_days_lower['normalized_score'])
        curr_score = 0
        curr_score_lw = 0
        while (score_diff < .01) and (score_diff_lw < .01):
            prev_score = np.mean(dna_days_upper['normalized_score'])
            prev_score_lw = np.mean(dna_days_lower['normalized_score'])
            # CREATE THE NEXT GENERATION AFTER INIT
            dna_days_upper = dna_days_upper.apply(self.create_next_gen, axis=1)
            dna_days_lower = dna_days_lower.apply(self.create_next_gen, axis=1)
            # RATE THE NEW GENERATION
            dna_days_upper['day_rating'] = dna_days_upper.apply(
                self.aggregated_day_rating, axis=1)

            dna_days_lower['day_rating'] = dna_days_lower.apply(
                self.aggregated_day_rating, axis=1)
            # NORMALIZE NEXT GEN SCORES
            self.normalize_day_rating(dna_days_upper)
            curr_score = np.mean(dna_days_upper['normalized_score'])

            self.normalize_day_rating(dna_days_lower)
            curr_score_lw = np.mean(dna_days_lower['normalized_score'])
            # CREATE THE MATING POOL OF NEXT GEN
            self.upper_mating_pool = self.generate_mating_pool(dna_days_upper)
            self.lower_mating_pool = self.generate_mating_pool(dna_days_lower)
            score_diff = curr_score - prev_score
            score_diff_lw = curr_score_lw - prev_score_lw
        

            """
            NOTE: NEXT STEPS: FROM THE DAYS CREATE USE THEM AS GENES FOR MICROCYCLES
            Combine the upper days and lower days into one gene pool
            Generate pheno types of microcycles using days as genes
            Rate microcycle, (include score from days)
            Mating Pool of Microcycles
            TODO: determine if its better to manually select days vs evolution 
            """
        # TODO: different functions for each microcycle length type (4,3,2) etc
        if self.usr_lvl == 'beginner' and self.no_days == 4:
            # return self.micro_4_evolution(m_pop_size, dna_days_upper, dna_days_lower)
            # NOTE: testing no dup days comparable
            upday_sec = self.no_dup_days_comparable(self.upper_mating_pool.iloc[[0]], self.upper_mating_pool)
        elif(self.usr_lvl == 'beginner' and self.no_days == 3):
            # TODO implement 3 day, 2_day
            return 
        else:
            return 'need to add more features - stick to just four day plans for now'
