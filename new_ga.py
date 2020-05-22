import numpy as np
import operator as op
import pandas as pd
from pandas.io.json import json_normalize
import pprint
import random
# import ga_fit_func as gff
import time
# import experimental_mod as exmod
from tqdm import tqdm as loadbar
from sklearn import preprocessing
from timeit import default_timer as timer
from ast import literal_eval

ex_gene_pool = None
usr_lvl = None
goal = None
no_days = None
no_exs = None

def create_total_genes(ex_len):
  if usr_lvl == 'beginner':
    ex_genes = ex_gene_pool[ex_gene_pool['level'] == 'beginner']
  elif usr_lvl == 'intermediate':
    ex_genes = ex_gene_pool[ex_gene_pool['level'] == 'intermediate']
  else:
    ex_genes = ex_gene_pool[ex_gene_pool['level'] == 'advanced']
  
  ex_df = ex_genes.sample(n=ex_len)

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


def create_up_genes(ex_len):
  #TODO: FURTHER TESTING NEEDED FOR SELECT STATEMENT
  up_ex_genes = ex_gene_pool[ex_gene_pool['muscle_group'] == 'upperbody']
  if usr_lvl == 'beginner':
    up_ex_genes = up_ex_genes[up_ex_genes['level'] == 'beginner']
  elif usr_lvl == 'intermediate':
    up_ex_genes = up_ex_genes[up_ex_genes['level'] == 'intermediate']
  else:
    up_ex_genes = up_ex_genes[up_ex_genes['level'] == 'advanced']

  ex_df = up_ex_genes.sample(n=ex_len)

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


def create_lw_genes(ex_len):
  lw_ex_genes = ex_gene_pool[ex_gene_pool['muscle_group'] == 'lowerbody']
  if usr_lvl == 'beginner':
    lw_ex_genes = lw_ex_genes[lw_ex_genes['level'] == 'beginner']
  elif usr_lvl == 'intermediate':
    lw_ex_genes = lw_ex_genes[lw_ex_genes['level'] == 'intermediate']
  else:
    lw_ex_genes = lw_ex_genes[lw_ex_genes['level'] == 'advanced']

  ex_df = lw_ex_genes.sample(n=ex_len)

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


def create_fb_day_pheno(ex_len):
  exercises = create_total_genes(ex_len)
  day = {'day':'','day_rating':0,'day_type':'fullbody', 'exercises':exercises, 'ex_l_len':ex_len,'goal':goal,'usr_lvl':usr_lvl}
  day_series = pd.Series(day)
  return day_series


def create_up_day_pheno(ex_len, usr_lvl, goal):
  exercises = create_up_genes(ex_len)
  day = {'day':'','day_rating':0,'day_type':'upperbody','exercises':exercises, 'ex_l_len':ex_len,'goal':goal,'usr_lvl':usr_lvl}
  day_series = pd.Series(day)
  return day_series


def create_lw_day_pheno(ex_len, usr_lvl, goal):
  exercises = create_lw_genes(ex_len)
  day = {'day': '', 'day_rating': 0, 'day_type': 'lowerbody',
         'exercises': exercises, 'ex_l_len': ex_len, 'goal': goal, 'usr_lvl': usr_lvl}
  day_series = pd.Series(day)
  return day_series


def generate_mating_pool(day_dna_df):
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


def generate_m_mating_pool(micro_dna):
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


def create_next_gen(day_series):
  if day_series['day_type'] == 'upperbody':
    parents_df = upper_mating_pool.sample(n=2)
  elif day_series['day_type'] == 'lowerbody':
    parents_df = lower_mating_pool.sample(n=2)
  else:
    # parents_df = mix_mating_pool.sample(n=2)
    print('TODO: add mixed mating pool')
  child = day_crossover(parents_df, day_series['ex_l_len'], day_series['usr_lvl'], day_series['goal'])
  child = day_mutation_chance(mutation_rate, child)
  return child  


def day_crossover(parent_df, ex_len, usr_lvl, goal):
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


def day_mutation_chance(mutation_rate, child):
  child_ex_df = pd.DataFrame(child['exercises'])
  child_ex_df['rand_chance'] = np.random.uniform(low=0,high=1,size=child_ex_df.shape[0])
  is_mutatable = child_ex_df['rand_chance'] < mutation_rate
  mutatable_children = child_ex_df[is_mutatable].copy()
  if not mutatable_children.empty:
    # print('BEFORE COMBINE FIRST IN MUTATION:::::')
    # print(child_ex_df)
    mutated_children = mutation(mutatable_children)
    # print('MUTATED CHILDREN ::::')
    # print(mutated_children)
    child_ex_df = mutated_children.combine_first(child_ex_df)
    # print('AFTER COMBINE FIRST IN MUTATION::::')
    # print(child_ex_df)
    # time.sleep(7)
    child['exercises'] = child_ex_df.to_dict('records')
  return child


def mutation(mutatable_df):
  mutated_df = pd.DataFrame(columns = mutatable_df.columns)
  mutated_df = mutated_df.append(ex_gene_pool.sample(n=len(mutatable_df)), ignore_index=True, sort=True)
  mutated_df.index = mutatable_df.index
  return mutated_df


# NOTE: VECTORIZED CASCADE FUNCTION
def validate_ex_cascade(ex_data, goal, usr_lvl):
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

#no duplicate exercises on same day types
def dup_check(ex_data):
  points = 0
  ex_data = pd.value_counts(ex_data['ex_name'].values, sort=False)
  ex_np_arr = np.array(ex_data)
  np.multiply(-10, ex_np_arr, out=ex_np_arr, where = ex_np_arr > 1)
  np.multiply(10, ex_np_arr, out=ex_np_arr, where = ex_np_arr == 1)
  points = ex_np_arr.sum()
  return points


def unqiue_type_check(ex_data):
  # TODO:TEST THIS
  # NOTE: NOT CURRENTLY USING, UNSURE IF NEEDED
  equipment = ex_data['ex_equipment']
  ex_data['type_equip'] = ex_data["ex_type"].str.cat(equipment, sep=", ")
  ex_data = pd.value_counts(ex_data['type_equip'].values, sort=False)
  ex_value_arr = np.array(ex_data)
  np.multiply(-10, ex_value_arr, out=ex_value_arr, where = ex_value_arr > 1)
  np.multiply(10, ex_value_arr, out=ex_value_arr, where = ex_value_arr == 1)
  return ex_value_arr.sum()


def aggregated_day_rating(day_series):
  ex_df = pd.DataFrame(day_series['exercises'])
  total_points = 0
  total_points += dup_check(ex_df)
  total_points += validate_ex_cascade(ex_df, day_series['goal'], day_series['usr_lvl'])
  # COMPLETED: reimplement fullbody, original code is using lists make it use np array
  if day_series.day_type == 'fullbody':
    # COMPLETED: reimplement this from old code
    total_points += rate_fbody_day(ex_df)
  return total_points


def normalize_day_rating(days_df):
  '''
  NOTE: CHANGED MIN TO -1990 IT USE TO BE -2050 BUT THE INDIVIDUALS DONT REACH THAT LOW
  NOTE: SECOND EDIT RANGE IS NOW -2800 TO 2050
  '''
  # min_max_norm = preprocessing.MinMaxScaler()
  # days_df['normalized_score'] = min_max_norm.fit_transform(
  #     days_df['day_rating'].values.reshape(-1, 1))
  # MIN AND MAX FOR 5 EX DAY
  fit_range = np.arange(-3550,2550)
  days_df['normalized_score'] = (days_df['day_rating'] - np.min(fit_range))/( np.max(fit_range) - np.min(fit_range))


def normalize_micro_rating(micro_df):
  # fit_range = np.arange(-16200, 11000)
  # used in microcycle gen 9,10,11
  # fit_range = np.arange(-4996,15000)
  # old fit range used for 8
  fit_range = np.arange(-30200,17400)
  micro_df['normalized_score'] = (micro_df['micro_rating'] - np.min(fit_range))/(np.max(fit_range) - np.min(fit_range))


def create_micro_pheno(no_days,usr_lvl, goal):
  working_days = microcyc_gene_pool.sample(n=no_days)
  # NOTE: added this conversion DATE: 10/31/2019
  working_days = working_days.to_dict('records')
  microcycle = {'micro_rating':'', 'workingdays': working_days, 'usr_lvl':usr_lvl, 'micro_type':goal}
  
  return pd.Series(microcycle)


def assess_4_day(wdays):
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


def assess_3_day(wdays):
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


def assess_2_day(wdays):
  pass


def rate_fbody_day(ex_df):
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


def no_dup_days(wdays):
  """
  For every day in the working days df
  combine all days exs into one array
  then count the number of times each ex in ex array appears
  they should all be unique.
  for ex len 5 = 50 and so on else -50
  """
  ex_arr = np.array([])
  for index,row in wdays.iterrows():
    ex_df = pd.DataFrame(row['exercises'], columns = ['ex_name','muscle_group','ex_mech_type','ex_type','ex_equipment','level','reps','sets','main-muscle-worked','movement_size','load_size'])
    ex_arr = np.append(ex_arr,ex_df['ex_name'].values)
  val_c = pd.value_counts(ex_arr, sort=False)
  arr_val_c = np.array(val_c)
  np.multiply(-400, arr_val_c, out=arr_val_c, where=arr_val_c > 1)
  np.multiply(200, arr_val_c, out=arr_val_c, where=arr_val_c == 1)
  # print('no dup days sum for 4 days ',arr_val_c.sum())
  return arr_val_c.sum()


def extract_exname(day_series):
  ex_df = pd.DataFrame(day_series['exercises'], columns = ['ex_name','muscle_group','ex_mech_type','ex_type','ex_equipment','level','reps','sets','main-muscle-worked','movement_size','load_size'])
  return ex_df['ex_name'].values

def no_d_days(wdays):
  ex_arr = np.array([])
  vals = wdays.apply(extract_exname,axis=1)
  ex_arr = np.append(vals.values)

def aggregated_micro_rating(micro_series):
  # print(micro_series['workingdays'].columns)
  if isinstance(micro_series['workingdays'], str):
    working_days = literal_eval(micro_series['workingdays'])
    print('converted to list')
    working_days = pd.DataFrame(working_days, columns = n_day_cols)
    print('now dataframe')
  else:
    working_days = pd.DataFrame(micro_series['workingdays'], columns = n_day_cols)
  total_points = 0
  if working_days.day_rating.values.sum() < 9000:
    total_points += -3000
  else:
    # NOTE ADDED ADDITIONAL POINTS
    total_points += 3500
  total_points += working_days.day_rating.values.sum()
  total_points += no_dup_days(working_days)
  if no_days == 4:
    total_points += assess_4_day(working_days)
  # print(total_points)
  return total_points


def create_next_gen_micro(micro_series):
  # TODO: finish this method add hyp and lean mass
  # if micro_series['micro_type'] == 'Strength':
  parents_df = micro_mating_pool.sample(n=2)
  
  child = micro_crossover(parents_df, usr_lvl, goal, no_days)
  child = micro_mutation_chance(mutation_rate, child)
  
  return child


def micro_crossover(parent_df, usr_lvl, goal, no_days):
  
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
  

def micro_mutation_chance(mutation_rate, child):
  child_micro_df = pd.DataFrame(child['workingdays'], columns = ['day','day_rating','day_type','exercises','ex_l_len','goal','usr_lvl','normalized_score','pop_num'])
  child_micro_df['rand_chance'] = np.random.uniform(low=0,high=1, size=child_micro_df.shape[0])
  is_mutatable = child_micro_df['rand_chance'] < mutation_rate
  
  mutatable_children = child_micro_df[is_mutatable].copy()
  
  if not mutatable_children.empty:
    mutated_children = mic_mutation(mutatable_children)
    child_micro_df = mutated_children.combine_first(child_micro_df)
    
    child['workingdays'] = child_micro_df.to_dict('records')
  
  return child


# TODO: MUTATION FUNCTION FOR MICROCYCLES
def mic_mutation(mutatable_df):
  
  mutated_df = pd.DataFrame(columns = mutatable_df.columns)
  mutated_df = mutated_df.append(microcyc_gene_pool.sample(n=len(mutatable_df)), ignore_index=True, sort=True)
  mutated_df.index = mutatable_df.index
  return mutated_df


def test_shit(micro_series):
  m_df = pd.DataFrame(micro_series['workingdays'], columns = ['day','day_rating','day_type','exercises','ex_l_len','goal','usr_lvl','normalized_score','pop_num'])
  
  print(m_df['exercises'])
  time.sleep(.3)
  
# TODO: find out if no_exs affects plan creation - dont want to cause errors
# TODO: for dev purposes import the cleaned genes csv, but need to move it to mongodb instance
def start_GA(usr_lvl, goal, no_days):
  global no_exs
  global ex_gene_pool 
  ex_gene_pool = pd.read_csv('cleaned_ex_genes.csv', index_col=0)
  ex_cols = ex_gene_pool.columns
  no_exs = 5
  mutation_rate = 0.01
  pop_size = 1000
  m_pop_size = 600
  day_cols = ['day', 'day_rating', 'day_type',
              'exercises', 'ex_l_len', 'goal', 'usr_lvl']
  # creation of days
  dna_days_upper = pd.DataFrame(index=range(pop_size), columns=day_cols)
  dna_days_lower = pd.DataFrame(index=range(pop_size), columns=day_cols)
  gen_tracker = pd.DataFrame(columns=['gen_num', 'score', 'norm_score'])
  gen_tracker_lw = pd.DataFrame(columns=['gen_num', 'score', 'norm_score'])

  # INIT DAY PHENO TYPES
  dna_days_upper = dna_days_upper.apply(
      lambda x: create_up_day_pheno(no_exs, usr_lvl, goal), axis=1)
  dna_days_lower = dna_days_lower.apply(
      lambda x: create_lw_day_pheno(no_exs, usr_lvl, goal), axis=1)

  # RATE THE INIT PHENOS
  dna_days_upper['day_rating'] = dna_days_upper.apply(
      aggregated_day_rating, axis=1)
  dna_days_lower['day_rating'] = dna_days_lower.apply(
      aggregated_day_rating, axis=1)

  # NORMALIZE THE DAY RATING
  normalize_day_rating(dna_days_upper)
  normalize_day_rating(dna_days_lower)
  dna_days_upper.to_csv('FIRST_UP_13_OG.csv')
  dna_days_lower.to_csv('FIRST_LW_13_OG.csv')

  # CREATE AN (INITIAL) MATING POOL BASED ON NORM SCORES
  # AND TRACKING THE GENERATION NUMBER RATINGS
  upper_mating_pool = generate_mating_pool(dna_days_upper)
  gen_num = 0
  gen_tracker['gen_num'] = np.full(pop_size, gen_num)
  gen_tracker['score'] = dna_days_upper['day_rating']
  gen_tracker['norm_score'] = dna_days_upper['normalized_score']

  lower_mating_pool = generate_mating_pool(dna_days_lower)
  gen_num_lw = 0
  gen_tracker_lw['gen_num'] = np.full(pop_size, gen_num_lw)
  gen_tracker_lw['score'] = dna_days_lower['day_rating']
  gen_tracker_lw['norm_score'] = dna_days_lower['normalized_score']

  score_diff = 0
  score_diff_lw = 0

  prev_score = np.mean(dna_days_upper['normalized_score'])
  prev_score_lw = np.mean(dna_days_lower['normalized_score'])
  curr_score = 0
  curr_score_lw = 0
  # TODO: REPLACE THIS WHILE LOOP WITH THE ONE FURTHER DOWN
  # GENERATIONS AFTER INIT STOP AT CONVERGENCE OF RATINGS
  while (score_diff < .01) and (score_diff_lw < .01):
    prev_score = np.mean(dna_days_upper['normalized_score'])
    prev_score_lw = np.mean(dna_days_lower['normalized_score'])
    gen_num += 1
    gen_num_lw += 1
    # CREATE THE NEXT GENERATION AFTER INIT
    dna_days_upper = dna_days_upper.apply(create_next_gen, axis=1)
    dna_days_lower = dna_days_lower.apply(create_next_gen, axis=1)

    # RATE THE NEW GENERATION
    dna_days_upper['day_rating'] = dna_days_upper.apply(
        aggregated_day_rating, axis=1)

    dna_days_lower['day_rating'] = dna_days_lower.apply(
        aggregated_day_rating, axis=1)

    # NORMALIZE NEXT GEN SCORES
    normalize_day_rating(dna_days_upper)
    curr_score = np.mean(dna_days_upper['normalized_score'])

    normalize_day_rating(dna_days_lower)
    curr_score_lw = np.mean(dna_days_lower['normalized_score'])

    # CREATE THE MATING POOL OF NEXT GEN
    upper_mating_pool = generate_mating_pool(dna_days_upper)
    lower_mating_pool = generate_mating_pool(dna_days_lower)

    score_diff = curr_score - prev_score
    score_diff_lw = curr_score_lw - prev_score_lw
    # concatenate the new gen's mating pool with the existing one
    # upper_mating_pool = pd.concat([upper_mating_pool, generate_mating_pool(dna_days_upper)], ignore_index=True, copy=False)
    # shuffle the matingpool inplace and reset the index
    # upper_mating_pool = upper_mating_pool.sample(frac=1).reset_index(drop=True)

    # PROGRESS TRACKING
    gen_df = pd.DataFrame(
        {'gen_num': np.full(pop_size, gen_num), 'score': dna_days_upper['day_rating'], 'norm_score': dna_days_upper['normalized_score']})
    gen_tracker = gen_tracker.append(gen_df, ignore_index=True)
  dna_days_upper.to_csv('LAST_UPPER_GEN_OG.csv')
  gen_tracker.to_csv('upper_generation_score_OG.csv')
  print('finished days ')
  gen_df_lw = pd.DataFrame(
      {'gen_num': np.full(pop_size, gen_num), 'score': dna_days_lower['day_rating'], 'norm_score': dna_days_lower['normalized_score']})
  gen_tracker_lw = gen_tracker_lw.append(gen_df_lw, ignore_index=True)
  dna_days_lower.to_csv('LAST_LOWER_GEN_OG.csv')
  gen_tracker_lw.to_csv('lower_generation_score_OG.csv')

  """
  NOTE: NEXT STEPS: FROM THE DAYS CREATE USE THEM AS GENES FOR MICROCYCLES
  Combine the upper days and lower days into one gene pool
  Generate pheno types of microcycles using days as genes
  Rate microcycle, (include score from days)
  Mating Pool of Microcycles
  """
  # THIS DATAFRAME WILL TRACK THE PROGRESS OF ALL GENERATIONS OF MICROCYCLES
  gen_m_tracker = pd.DataFrame(columns=['gen_num', 'score', 'norm_score'])

  if usr_lvl == 'beginner' and no_days == 4:
    print('starting microcycle')
    micro_cols = ['micro_rating', 'workingdays', 'usr_lvl', 'micro_type']
    n_day_cols = ['day', 'day_rating', 'day_type', 'exercises',
                  'ex_l_len', 'goal', 'usr_lvl', 'normalized_score', 'pop_num']
    dna_microcycles = pd.DataFrame(
        index=range(m_pop_size), columns=micro_cols)

    microcyc_gene_pool = dna_days_upper.append(
        dna_days_lower, ignore_index=True)
    # only choose days that are moderate to high score.
    microcyc_gene_pool = microcyc_gene_pool[microcyc_gene_pool['day_rating'] > 2000]
    dna_microcycles = dna_microcycles.apply(
        lambda x: create_micro_pheno(no_days, usr_lvl, goal), axis=1)
    dna_microcycles['micro_rating'] = dna_microcycles.apply(
        aggregated_micro_rating, axis=1)

    # COMPLETED: normalize microcyc score
    normalize_micro_rating(dna_microcycles)
    # COMPLETED: generate mating pool for microcyc
    micro_mating_pool = generate_m_mating_pool(dna_microcycles)
    # GENERATION NUMBER OF EACH MICROCYCLE GENERATION
    m_gen_num = 0
    score_diff = 0
    prev_score = np.mean(dna_microcycles['normalized_score'])
    curr_score = 0
    # POPULATING INITIAL COLUMNS OF TRACKER
    gen_m_tracker['gen_num'] = np.full(m_pop_size, m_gen_num)
    gen_m_tracker['score'] = dna_microcycles['micro_rating']
    gen_m_tracker['norm_score'] = dna_microcycles['normalized_score']

    # while(score_diff < .1):
    for i in loadbar(range(1600)):
      m_gen_num += 1
      # prev_score = np.mean(dna_microcycles['normalized_score'])
      dna_microcycles = dna_microcycles.apply(create_next_gen_micro, axis=1)

      dna_microcycles['micro_rating'] = dna_microcycles.apply(
          aggregated_micro_rating, axis=1)
      normalize_micro_rating(dna_microcycles)
      # curr_score = np.mean(dna_microcycles['normalized_score'])

      micro_mating_pool = generate_m_mating_pool(dna_microcycles)
      # score_diff = curr_score - prev_score
      # UPDATE TRACKER WITH NEXT GENERATION
      m_gen_df = pd.DataFrame({'gen_num': np.full(
          m_pop_size, m_gen_num), 'score': dna_microcycles['micro_rating'], 'norm_score': dna_microcycles['normalized_score']})
      gen_m_tracker = gen_m_tracker.append(m_gen_df, ignore_index=True)

    gen_m_tracker.to_csv('microcycles_generations_14.csv')
    print('finished microcycle')
    dna_microcycles.to_csv('MICROCYCLES_OG_14.csv')

      



