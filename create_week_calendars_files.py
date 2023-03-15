import copy
import numpy as np
import pandas as pd
import math
from random import shuffle
from random import randint

import assign_week_to_calendar


def normalize_points(preferations_df):
    """
    turn negative points to zeros.
    :param preferations_df:
    :return:
    """
    # rows 5 to 35 contain the points
    row_sum = preferations_df.iloc[:, 5:36].sum(axis=1).to_numpy()[:, np.newaxis]
    preferations_df.iloc[:, 5:36] = np.divide(preferations_df.iloc[:, 5:36], row_sum)


def create_preferations_df(preferations_df_path):
    preferations_df = pd.read_excel(preferations_df_path)
    preferations_df["number_of_shifts"] = np.zeros(preferations_df.shape[0])
    preferations_df["total_points"] = np.zeros(preferations_df.shape[0])
    normalize_points(preferations_df)
    return preferations_df


def create_empty_calendar_test(number_of_days_in_month=5, first_day_of_month=0):
    days_of_the_week = ["sunday", "monday", "tuesday", "wednesday", "thursday"]  # "friday", "saturday"]
    week_number = np.arange(1).astype(int)
    current_day_of_month = 0
    current_day_of_week = first_day_of_month
    empty_calendar = pd.DataFrame(columns=days_of_the_week, index=week_number)
    for i in week_number:
        # we use np.nan end not None since one can't use np.isnan on a None value
        appended_days = [[j, np.nan] for j in range(current_day_of_week, number_of_days_in_month)]
        empty_calendar.iloc[i, current_day_of_week:number_of_days_in_month] = appended_days
        current_day_of_week = 0
        current_day_of_month = current_day_of_month + len(appended_days)
    return empty_calendar


def create_empty_calendar(number_of_days_in_month, first_day_of_month):
    """
    a dataframe whose elements are lists cannot be deeply copied. hence the use
     of tuples.
     `None`` was used since pd. NA has an ambiguous boolean value.
    :param number_of_days_in_month: int, 28 <= first_day_of_month <= 31
    :param first_day_of_month: int, 0 <= first_day_of_month <= 6
    :return:
    """
    days_of_the_week = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
    week_number = np.arange(6).astype(int)
    current_day_of_month = 1
    current_day_of_week = first_day_of_month
    empty_calendar = pd.DataFrame(columns=days_of_the_week, index=week_number)
    for i in week_number:
        # we use np.nan end not None since one can't use np.isnan on a None value
        appended_days = [[j, np.nan] for j in range(current_day_of_week, 7)]
        empty_calendar.iloc[i, current_day_of_week:7] = appended_days
        current_day_of_week = 0
        current_day_of_month = current_day_of_month + len(appended_days)
    # emptying days will always happen at the last week
    if current_day_of_month > number_of_days_in_month:
        number_of_days_to_delete = current_day_of_month - number_of_days_in_month - 1
        if number_of_days_to_delete <= 7:
            empty_calendar.iloc[-1, 7 - number_of_days_to_delete:7] = \
                [np.nan for _ in range(number_of_days_to_delete)]
        else:
            empty_calendar.iloc[-1] = [np.nan for _ in range(7)]
            empty_calendar.iloc[-2, 14 - number_of_days_to_delete:7] = \
                [np.nan for _ in range(number_of_days_to_delete - 7)]
    return empty_calendar


def calculate_first_and_last_day_of_week(next_week_calendar):
    first_day_of_week = 0
    last_day_of_week = 6
    while (first_day_of_week < 6) and (np.all(np.isnan(next_week_calendar[first_day_of_week]))):
        first_day_of_week += 1
    while (last_day_of_week > 0) and (np.all(np.isnan(next_week_calendar[last_day_of_week]))):
        last_day_of_week -= 1
    return first_day_of_week, last_day_of_week


def there_are_extra_interns(amount_of_interns, min_num_of_shifts_per_week, num_of_days_in_week):
    # some weeks in a month have less than 7 days
    return (amount_of_interns * min_num_of_shifts_per_week) > num_of_days_in_week


def create_zero_to_n_combinations_list(result_list, element_list, smallest_element, largest_element,
                                       amount_of_elements):
    """
    the combinations list is a rising series from element_list[0] to n.
    """
    if amount_of_elements <= 0:
        result_list.append(element_list)
        return None
    # to prevent duplicates list [1,2] and [2,1]
    possible_elements_to_add = range(smallest_element, largest_element + 1)
    for i in possible_elements_to_add:
        if smallest_element == 0:
            element_list = [i]
            create_zero_to_n_combinations_list(result_list, element_list, smallest_element=element_list[-1] + 1,
                                               largest_element=largest_element,
                                               amount_of_elements=amount_of_elements - 1)
        else:
            new_element_list = list(element_list)
            new_element_list.append(i)
            create_zero_to_n_combinations_list(result_list, new_element_list, smallest_element=new_element_list[-1] + 1,
                                               largest_element=largest_element,
                                               amount_of_elements=amount_of_elements - 1)
    # didn't manage to add enough elements
    return result_list


def create_extra_interns_list(amount_of_interns, min_num_of_shifts_per_week, number_of_days_in_week):
    """
    possible combinations of extra interns indices.
    this list should be calculated once each week, since amount of days in week is not a constant.
    IMPORTANT: number_of_days_in_week should be >0.
    :param min_num_of_shifts_per_week:
    :param amount_of_interns:
    :param number_of_days_in_week:
    :return:
    """
    if not there_are_extra_interns(amount_of_interns, min_num_of_shifts_per_week, number_of_days_in_week):
        return []
    else:
        # do ceiling value since the last interns can do shifts, but less than the minimum.
        amount_of_extra_interns = math.ceil((amount_of_interns * min_num_of_shifts_per_week -
                                             number_of_days_in_week) / min_num_of_shifts_per_week)
        extra_interns_list = create_zero_to_n_combinations_list(
            result_list=[], element_list=None, smallest_element=0, largest_element=amount_of_interns - 1,
            amount_of_elements=amount_of_extra_interns)
        # current_extra_interns = \
        #     interns_permutation[-(len(interns_permutation) * min_num_of_shifts_per_week - number_of_days_in_week):]
        # current_extra_interns.sort()
        # extra_interns_list.remove(current_extra_interns)
        return extra_interns_list


def return_next_permutation(interns_permutation, min_num_of_shifts_per_week, number_of_days_in_week,
                            extra_interns_list):
    """
    should try all combinations of interns to be extra (not circular change)
    :return:
    """
    # todo: make sure that in the first and last weeks there are different interns, to prevent unfairness.
    num_of_interns = len(interns_permutation)
    # create a copy of interns_permutations, else will affect following iterations of return_best_valid_current_week.
    interns_permutation = copy.deepcopy(interns_permutation)
    if there_are_extra_interns(num_of_interns, min_num_of_shifts_per_week, number_of_days_in_week):
        if not extra_interns_list:
            return None
        # choose extra-interns that haven't played that role, randomly.
        random_extra_interns_index = randint(0, len(extra_interns_list) - 1)
        extra_interns = extra_interns_list[random_extra_interns_index]
        non_extra_interns = [element for element in interns_permutation if
                             (element in interns_permutation) and (element not in extra_interns)]
        shuffle(non_extra_interns)
        non_extra_interns.extend(extra_interns)
        interns_permutation = non_extra_interns
        extra_interns_list.remove(extra_interns)
    # if there are no extra interns - shuffle
    else:
        shuffle(interns_permutation)
    return interns_permutation


def create_initial_interns_permutation(number_of_interns):
    interns_permutation = [i for i in range(number_of_interns)]
    shuffle(interns_permutation)
    return interns_permutation


def calculate_num_of_days_in_week(calendar_to_fill, week_number):
    return calendar_to_fill.iloc[week_number].count()


def return_current_week_calendars(current_calendar, week_number, preferations_df,
                                  interns_permutation, max_num_of_shifts_per_week, min_num_of_shifts_per_week,
                                  last_day_of_week, first_day_of_week,
                                  current_week_calendars_were_calculated):
    num_of_days_in_week = calculate_num_of_days_in_week(current_calendar, week_number)
    # print(current_calendar)
    print("num of days in week", week_number, "days", num_of_days_in_week)
    # the implementation of the load and save features into the program
    if not current_week_calendars_were_calculated:
        current_week_calendars = assign_week_to_calendar.return_valid_weekly_calendars(
            preferations_df=preferations_df,
            interns_permutation=interns_permutation,
            max_num_of_shifts=max_num_of_shifts_per_week,
            min_num_of_shifts=min_num_of_shifts_per_week,
            number_of_days_in_month=last_day_of_week + 1 - first_day_of_week,
            current_week=pd.DataFrame(current_calendar.iloc[week_number, :]).transpose().to_numpy().tolist(),
            first_day_of_month=first_day_of_week,
            last_day_in_week=last_day_of_week)
        np.save(file=f"week_calendars{week_number}", arr=current_week_calendars, allow_pickle=True, fix_imports=False)
    else:
        current_week_calendars = np.load(file=f"week_calendars{week_number}.npy", mmap_mode=None, allow_pickle=True,
                                         fix_imports=False, encoding="ASCII")
    return current_week_calendars


def create_week_calendars_as_numpy_file(preferations_df, max_num_of_shifts_per_week,
                                        min_num_of_shifts_per_week, number_of_days_in_month,
                                        first_day_of_month, week_number, first_day_of_week, last_day_of_week,
                                        num_of_days_in_week,
                                        interns_permutation):
    """
    use best_week_calendars_list for statistical analysis.

    min_num_of_shifts_per_week == max_num_of_shifts_per_week == 1. if 0 or >1 it takes to long to calculate. do this
    adjustment on the output of this subroutine.
    
    :return:
    """  # todo: change to per week.
    calendar_to_fill = create_empty_calendar(number_of_days_in_month, first_day_of_month)
    # to calculate and np.save or to np.load current_week_calendars
    current_week_calendars_were_calculated = False
    # bool(int(input("enter 0 if calendars have not already been calculated, and 1 otherwise")))
    if num_of_days_in_week <= 0:  # the last week of the month might have 0 days
        return None
    current_week_calendars = \
        return_current_week_calendars(calendar_to_fill, week_number, preferations_df,
                                  interns_permutation, max_num_of_shifts_per_week, min_num_of_shifts_per_week,
                                  last_day_of_week, first_day_of_week,
                                  current_week_calendars_were_calculated)
    return current_week_calendars
