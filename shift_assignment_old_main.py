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


# we use a while-loop instead of a for-loop because the number of
#  iterations depends also on the bans lists, so it's hard to calculate
def calendar_week_is_full(calendar):
    for i in range(len(calendar[0])):  # calendar shape is (1, 7, 2)
        # not a day which does not exist is month
        if not np.all(np.isnan(np.array(calendar[0][0][i]))):
            if np.any(np.isnan(np.array(calendar[0][0][i][1]))):
                return False
    return True


def sunday_shift_after_saturday_shift(best_week_calendar, week_number, current_calendar):
    if week_number > 0:
        # if the new week's sunday is not empty
        if not np.all(np.isnan(best_week_calendar[0][0][0])):
            # if the new week's sunday and the previous' week's saturday shifts
            # are done by the same person
            if best_week_calendar[0][0][0][1] == current_calendar.iloc[week_number - 1, 6][1]:
                return True
    return False


def month_num_shifts_exceed_upper_bound(max_num_of_shifts, monthly_interns_num_shifts_array, best_week_calendar):
    return np.any((monthly_interns_num_shifts_array + best_week_calendar[1][1]) > max_num_of_shifts)


def month_num_shifts_under_lower_bound_in_last_week(min_num_of_shifts, monthly_interns_num_shifts_array,
                                                    best_week_calendar):
    return np.any((monthly_interns_num_shifts_array + best_week_calendar[1][1]) < min_num_of_shifts)


def calculate_current_calendar_difference(best_week_calendars_list, number_of_participants):
    current_calendar_shifts_array = np.zeros(number_of_participants)
    for week_calendar in best_week_calendars_list:
        current_calendar_shifts_array += week_calendar[1][1]
    return current_calendar_shifts_array


def num_shifts_difference(best_week_calendar, best_week_calendars_list):
    # only aggregated. todo: also for the current week? too specific.... maybe not needed also for aggregated...
    best_calendar_num_shifts_array = best_week_calendar[1][1]
    current_calendar_shifts_array = calculate_current_calendar_difference(best_week_calendars_list,
                                                                          best_week_calendar[1][1].shape[0])
    total_num_shifts_array = best_calendar_num_shifts_array + current_calendar_shifts_array
    return np.max(total_num_shifts_array) - np.min(total_num_shifts_array)


def points_distribution_is_fair(best_week_calendar, best_week_calendars_list):
    # in best_week_calendar
    desired_maximal_variance = float(input("enter desired maximal points variance between interns"))
    # aggregated
    # distribution is fair
    # variance is not too high
    return True


def shifts_distribution_is_fair(current_week_calendars, best_week_calendar, current_calendar,
                                interns_num_shifts_array, interns_points_array, best_week_calendars_list):
    # aggregated num shifts distribution/variance - max of 1 difference. fixme: it's a heuristic
    if num_shifts_difference(best_week_calendar, best_week_calendars_list) > 1:
        return False
    # points distribution/variance - for best_week_calendar + aggregated
    if not points_distribution_is_fair(best_week_calendar, best_week_calendars_list):
        return False
    return True
    # print best 10 for each category:
    # total points, uniform points dist., uniform num. shifts dist.
    # lowest_points_calendars = np.array(main_calendars[2]).sort(axis=0)[:10]
    # print(np.array(main_calendars)[:][2])
    # lowest_points_calendars_indices = np.where(main_calendars == lowest_points_calendars)
    # lowest_points_calendars = main_calendars[lowest_points_calendars_indices]


def best_week_calendar_is_valid(current_week_calendars, best_week_calendar, current_calendar,
                                week_number, min_num_of_shifts, max_num_of_shifts, monthly_interns_num_shifts_array,
                                monthly_interns_points_array, best_week_calendars_list, is_last_week=False):
    """
    monthly_interns_num_shifts_array and monthly_interns_points_array do not include the current week.

    due to handling current_week_calendars by partitions by no. of holes, no need to check whether calendar_is_full
    """
    # todo: test a positive case (with collision)
    if sunday_shift_after_saturday_shift(best_week_calendar, week_number, current_calendar):
        return False
    # todo: test
    if month_num_shifts_exceed_upper_bound(max_num_of_shifts, monthly_interns_num_shifts_array, best_week_calendar):
        return False
    # todo: test
    if is_last_week and month_num_shifts_under_lower_bound_in_last_week(min_num_of_shifts,
                                                                        monthly_interns_num_shifts_array,
                                                                        best_week_calendar):
        return False
    # todo: test. here use a dash app
    if not shifts_distribution_is_fair(current_week_calendars, best_week_calendar, current_calendar,
                                       monthly_interns_num_shifts_array, monthly_interns_points_array,
                                       best_week_calendars_list):
        return False
    return True


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


def create_extra_interns_list(interns_permutation, min_num_of_shifts_per_week, number_of_days_in_week):
    """
    possible combinations of extra interns indices.
    this list should be calculated once each week, since amount of days in week is not a constant.
    IMPORTANT: number_of_days_in_week should be >0.
    :param min_num_of_shifts_per_week:
    :param interns_permutation:
    :param number_of_days_in_week:
    :return:
    """
    if not there_are_extra_interns(len(interns_permutation), min_num_of_shifts_per_week, number_of_days_in_week):
        return interns_permutation
    else:
        # do ceiling value since the last interns can do shifts, but less than the minimum.
        amount_of_extra_interns = math.ceil((len(interns_permutation)*min_num_of_shifts_per_week -
                                            number_of_days_in_week) / min_num_of_shifts_per_week)
        extra_interns_list = create_zero_to_n_combinations_list(
            result_list=[], element_list=None, smallest_element=0, largest_element=len(interns_permutation)-1,
            amount_of_elements=amount_of_extra_interns)
        current_extra_interns = \
            interns_permutation[-(len(interns_permutation) * min_num_of_shifts_per_week - number_of_days_in_week):]
        current_extra_interns.sort()
        extra_interns_list.remove(current_extra_interns)
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


def return_current_week_calendars(current_calendar, week_number, preferations_df, min_num_of_shifts,
                                  interns_permutation, max_num_of_shifts_per_week, min_num_of_shifts_per_week,
                                  last_day_of_week, first_day_of_week, extra_interns_list,
                                  current_week_calendars_were_calculated):
    num_of_days_in_week = calculate_num_of_days_in_week(current_calendar, week_number)
    # print(current_calendar)
    print("num of days in week", week_number, "days", num_of_days_in_week)
    # the implementation of the load and save features into the program
    if not current_week_calendars_were_calculated:
        if there_are_extra_interns(preferations_df.shape[0], min_num_of_shifts, num_of_days_in_week):
            # each week change interns permutation for fairness
            interns_permutation = interns_permutation if (week_number == 0) \
                else return_next_permutation(interns_permutation, min_num_of_shifts_per_week, num_of_days_in_week,
                                             extra_interns_list)
            current_week_calendars = assign_week_to_calendar.return_valid_weekly_calendars(
                preferations_df=preferations_df,
                interns_permutation=interns_permutation,
                max_num_of_shifts=max_num_of_shifts_per_week,
                min_num_of_shifts=min_num_of_shifts_per_week,
                number_of_days_in_month=last_day_of_week + 1 - first_day_of_week,
                current_week=pd.DataFrame(current_calendar.iloc[week_number, :]).transpose().to_numpy().tolist(),
                first_day_of_month=first_day_of_week,
                last_day_in_week=last_day_of_week)
        else:
            interns_permutation = range(preferations_df.shape[0])
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


def calculate_week_calendar_num_holes(calendar):
    calendar_num_holes = 0
    for i in range(len(calendar[0])):  # calendar shape is (1, 7, 2)
        # not a day which does not exist is month
        if not np.all(np.isnan(np.array(calendar[0][0][i]))):
            if np.isnan(np.array(calendar[0][0][i][1])):
                calendar_num_holes += 1
    return calendar_num_holes


def partition_current_week_calendars_by_num_holes(current_week_calendars):
    current_week_calendars_partitioned_by_num_holes = [[], [], [], [], [], [], [], []]  # 0 <= num_holes <= 7
    for calendar in current_week_calendars:
        calendar_num_holes = calculate_week_calendar_num_holes(calendar)
        current_week_calendars_partitioned_by_num_holes[calendar_num_holes].append(calendar)
    return current_week_calendars_partitioned_by_num_holes


def return_best_valid_current_week(current_calendar, preferations_df, week_number, max_num_of_shifts_per_week,
                                   min_num_of_shifts_per_week, last_day_of_week, first_day_of_week, min_num_of_shifts,
                                   max_num_of_shifts, best_week_calendars_list, number_of_interns, interns_permutation,
                                   extra_interns_list, monthly_interns_num_shifts_array, monthly_interns_points_array,
                                   current_week_calendars_were_calculated):
    """
    calculate expectation and variance of `current_week_calendars`
    plot these statistics' distributions.

    check total max num of shifts is not exceeded.

    in current_week_calendars each element is
    array[list([day, intern_index]...), list(interns_points_array, interns_num_shifts_array), total_points]

    a lower score is better - think like a loss function.
    :return:
    """
    current_week_calendars = \
        return_current_week_calendars(current_calendar, week_number, preferations_df, min_num_of_shifts,
                                      interns_permutation, max_num_of_shifts_per_week, min_num_of_shifts_per_week,
                                      last_day_of_week, first_day_of_week, extra_interns_list,
                                      current_week_calendars_were_calculated)
    # from those take all calendars with least un-occupied shifts (preferably full)
    # replaces calendar_week_is_full
    current_week_calendars_partitioned_by_num_holes = partition_current_week_calendars_by_num_holes(
        current_week_calendars)
    # for each partition, from 0 to 7
    for current_week_calendars_partition in current_week_calendars_partitioned_by_num_holes:
        current_week_calendars_partition = np.array(current_week_calendars_partition, dtype=object)
        # return the lowest score calendar which is full
        min_score_index = np.argmin(current_week_calendars_partition[:, 2])
        best_week_calendar = current_week_calendars_partition[min_score_index, :]
        is_last_week = (current_calendar.shape[0] - 1 == week_number)
        # while we need to keep searching for the best calendar: not full or more calendars to check
        while not best_week_calendar_is_valid(current_week_calendars_partition, best_week_calendar, current_calendar,
                                              week_number, min_num_of_shifts, max_num_of_shifts,
                                              monthly_interns_num_shifts_array, monthly_interns_points_array,
                                              best_week_calendars_list, is_last_week):
            if current_week_calendars_partition.shape[0] > 1:
                # remove calendar from calendars array
                current_week_calendars_partition = np.delete(arr=current_week_calendars_partition, obj=min_score_index,
                                                             axis=0)
                # get best calendar from remaining calendars
                min_score_index = np.argmin(current_week_calendars_partition[:, 2])
                best_week_calendar = current_week_calendars_partition[min_score_index, :]
            else:  # no valid week-calendar exists
                break  # try next partition (some shifts might be empty)
        return best_week_calendar
    return None  # in all partitions no valid calendar


def update_according_to_valid_best_week_calendar(calendar_to_fill, week_number, best_week_calendar,
                                                 best_week_calendars_list, monthly_interns_num_shifts_array,
                                                 monthly_interns_points_array):
    calendar_to_fill.iloc[week_number, :] = best_week_calendar[0][0]
    best_week_calendars_list.append(best_week_calendar)
    # update aggregated num_shifts and points
    monthly_interns_num_shifts_array += best_week_calendar[1][1]
    monthly_interns_points_array += best_week_calendar[1][1]
    return calendar_to_fill, best_week_calendars_list, monthly_interns_num_shifts_array, monthly_interns_points_array


def should_stop_search_for_best_week_calendar(calendar_to_fill, week_number, preferations_df, min_num_of_shifts,
                                              extra_interns_list, interns_permutation, min_num_of_shifts_per_week):
    num_of_days_in_week = calculate_num_of_days_in_week(calendar_to_fill, week_number)
    if there_are_extra_interns(preferations_df.shape[0], min_num_of_shifts, num_of_days_in_week):
        # check if all interns permutations have been tried
        if not extra_interns_list:  # all useful permutations hve been tried
            print("all useful permutations with extra interns have been tried but best_week_calendar is "
                  "none - break. ", "week no. is ", week_number)
            return True, interns_permutation
        else:  # try next permutation since the current one didn't work
            interns_permutation = return_next_permutation(
                interns_permutation, min_num_of_shifts_per_week, num_of_days_in_week, extra_interns_list)
    else:  # since there are no extra interns there is no need to try other permutations.
        print("best_week_calendar is none and not extra interns - break. ", "week no. is ", week_number)
        return True, interns_permutation
    return False, interns_permutation


def handle_current_week_assignment(preferations_df, max_num_of_shifts_per_week, max_num_of_shifts,
                                   min_num_of_shifts_per_week, min_num_of_shifts, calendar_to_fill,
                                   week_number, last_day_of_week, first_day_of_week, best_week_calendars_list,
                                   extra_interns_list, monthly_interns_num_shifts_array, monthly_interns_points_array,
                                   interns_permutation, best_week_calendar,
                                   current_week_calendars_were_calculated):
    """
    to create a different useful permutation each time best_week_calendar is None, we calculate all the
    different combinations of interns to be extra (a subset of the power set, of all list of a certain
    length)  - see `else` clause, count no. useful permutations
    """
    # loop will break either if best_week_calendar is not None, or if all useful permutation have been tried.
    while best_week_calendar is None:  # next try hopefully the permutation will be different
        # insert the best(s) current week
        # todo: choose multiple best calendars - NP-hard
        best_week_calendar = \
            return_best_valid_current_week(calendar_to_fill, preferations_df, week_number,
                                           max_num_of_shifts_per_week,
                                           min_num_of_shifts_per_week, last_day_of_week, first_day_of_week,
                                           min_num_of_shifts, max_num_of_shifts, best_week_calendars_list,
                                           preferations_df.shape[0], interns_permutation, extra_interns_list,
                                           monthly_interns_num_shifts_array, monthly_interns_points_array,
                                           current_week_calendars_were_calculated)
        if best_week_calendar is not None:
            calendar_to_fill, best_week_calendars_list, monthly_interns_num_shifts_array, monthly_interns_points_array\
                = update_according_to_valid_best_week_calendar(calendar_to_fill, week_number, best_week_calendar,
                                                               best_week_calendars_list,
                                                               monthly_interns_num_shifts_array,
                                                               monthly_interns_points_array)
        else:
            print("a best calendar in week ", week_number, "was none ")
            stop_search_for_best_week_calendar, interns_permutation =\
                should_stop_search_for_best_week_calendar(calendar_to_fill, week_number, preferations_df,
                                                          min_num_of_shifts, extra_interns_list, interns_permutation,
                                                          min_num_of_shifts_per_week)
            if stop_search_for_best_week_calendar:
                break
    return (calendar_to_fill, best_week_calendars_list, monthly_interns_num_shifts_array,
            monthly_interns_points_array, interns_permutation, best_week_calendar)


def create_full_calendar(preferations_df, max_num_of_shifts_per_week, max_num_of_shifts, min_num_of_shifts_per_week,
                         min_num_of_shifts, number_of_days_in_month, first_day_of_month):
    """

    extra_interns_list is utilized only if there are extra interns, but since it's in a called subroutine, we will
    compute it anyway each week to avoid a warning.

    use best_week_calendars_list for statistical analysis.

    min_num_of_shifts_per_week == max_num_of_shifts_per_week == 1. if 0 or >1 it takes to long to calculate. do this
    adjustment on the output of this subroutine.
    :return:
    """
    # local variables initiation based on the subroutine's input
    calendar_to_fill = create_empty_calendar(number_of_days_in_month, first_day_of_month)
    first_day_of_week, last_day_of_week = calculate_first_and_last_day_of_week(calendar_to_fill.iloc[0, :])
    best_week_calendars_list = []
    # to calculate and np.save or to np.load current_week_calendars
    current_week_calendars_were_calculated = \
        bool(int(input("enter 0 if calendars have not already been calculated, and 1 otherwise")))
    print("my boolean ", current_week_calendars_were_calculated)
    # is zeroed only here
    monthly_interns_num_shifts_array = np.zeros(preferations_df.shape[0])
    monthly_interns_points_array = np.zeros(preferations_df.shape[0])
    # to make up for not allowing 0 shifts while there are too many interns, try different interns permutations
    # to make sure that each week different extra interns are last and thus do not do any shifts.
    for week_number in range(calendar_to_fill.shape[0]):
        interns_permutation = create_initial_interns_permutation(preferations_df.shape[0])
        best_week_calendar = None
        num_of_days_in_week = calculate_num_of_days_in_week(calendar_to_fill, week_number)
        if num_of_days_in_week <= 0:  # the last week of the month might have 0 days
            continue
        extra_interns_list = create_extra_interns_list(interns_permutation, min_num_of_shifts_per_week,
                                                       num_of_days_in_week)
        calendar_to_fill, best_week_calendars_list, monthly_interns_num_shifts_array, monthly_interns_points_array, \
            interns_permutation, best_week_calendar =\
            handle_current_week_assignment(preferations_df, max_num_of_shifts_per_week, max_num_of_shifts,
                                           min_num_of_shifts_per_week, min_num_of_shifts, calendar_to_fill, week_number,
                                           last_day_of_week, first_day_of_week, best_week_calendars_list,
                                           extra_interns_list, monthly_interns_num_shifts_array,
                                           monthly_interns_points_array, interns_permutation, best_week_calendar,
                                           current_week_calendars_were_calculated)
        # prepare for next week calculation
        if week_number < calendar_to_fill.shape[0] - 1:
            first_day_of_week, last_day_of_week = \
                calculate_first_and_last_day_of_week(calendar_to_fill.iloc[week_number + 1, :])
    return calendar_to_fill


if __name__ == '__main__':
    main_preferations_df = create_preferations_df("preferations_df.xlsx")
    main_preferations_df = main_preferations_df  # .iloc[0:2]
    main_preferations_df.loc[6] = pd.Series(main_preferations_df.iloc[5])
    main_preferations_df.loc[7] = pd.Series(main_preferations_df.iloc[5])
    main_first_day_of_month = 1  # monday
    main_last_day_of_week = 2
    main_preferations_df.iloc[:, 1:5] = -1  # disabling bans list
    # min_num_of_shifts_per_week must be >= 1 to allow good runtimes
    # todo: # add as input all current_week_calendars, to integrate with dash.
    main_calendars = create_full_calendar(preferations_df=main_preferations_df,
                                          max_num_of_shifts_per_week=1,
                                          max_num_of_shifts=4,
                                          min_num_of_shifts_per_week=1,
                                          min_num_of_shifts=4,
                                          number_of_days_in_month=30,
                                          first_day_of_month=main_first_day_of_month)
    # assign each not-full calendar one shift of each intern at the time - re-check fullness.
    print(main_calendars)
# todo: also handle a case of not having enough interns - still unoccupied shifts after assignment.
#  use calendar_week_is_full for that purpose
