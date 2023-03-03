import numpy as np

# create_empty_calendar is used to create the final calendar
# calculate_num_of_days_in_week is used to know if we are in a week with 0 days, and whether there are extra interns
# there_are_extra_interns is used to know whether should we calculate the numpy files again, with different permutations
# the extra_interns_list is used to create all useful permutations, for return_next_permutation.
# return_current_week_calendars, return_next_permutation are used in case we should re-calculate current_week_calendars,
# with a different interns permutation.
# create_week_calendars_as_numpy_files for re-calculation of files, with a different permutation.
from create_week_calendars_files import create_empty_calendar, calculate_num_of_days_in_week, there_are_extra_interns, \
    create_extra_interns_list, return_current_week_calendars, return_next_permutation,\
    create_week_calendars_as_numpy_file


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


def points_distribution_is_fair(best_week_calendar, best_week_calendars_list, input_statistics):
    # input_statistics includes variance
    print("got statistics")
    # in best_week_calendar
    # desired_maximal_variance = float(input("enter desired maximal points variance between interns"))
    # aggregated
    # distribution is fair
    # variance is not too high
    return True


def shifts_distribution_is_fair(current_week_calendars, best_week_calendar, current_calendar,
                                interns_num_shifts_array, interns_points_array, best_week_calendars_list,
                                input_statistics):
    # aggregated num shifts distribution/variance - max of 1 difference. fixme: it's a heuristic
    if num_shifts_difference(best_week_calendar, best_week_calendars_list) > 1:
        return False
    # points distribution/variance - for best_week_calendar + aggregated
    if not points_distribution_is_fair(best_week_calendar, best_week_calendars_list, input_statistics):
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
                                monthly_interns_points_array, best_week_calendars_list, input_statistics,
                                is_last_week=False):
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
                                       best_week_calendars_list, input_statistics):
        return False
    return True


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


def return_best_valid_current_week(current_calendar, week_number, min_num_of_shifts,
                                   max_num_of_shifts, best_week_calendars_list, number_of_interns, interns_permutation,
                                   monthly_interns_num_shifts_array, monthly_interns_points_array,
                                   current_week_calendars, input_statistics):
    """
    best_week_calendar = \
            return_best_valid_current_week(calendar_to_fill, week_number,
                                           min_num_of_shifts, max_num_of_shifts, best_week_calendars_list,
                                           preferations_df.shape[0], interns_permutation,
                                           monthly_interns_num_shifts_array, monthly_interns_points_array,
                                           current_week_calendars_were_calculated)



    calculate expectation and variance of `current_week_calendars`
    plot these statistics' distributions.

    check total max num of shifts is not exceeded.

    in current_week_calendars each element is
    array[list([day, intern_index]...), list(interns_points_array, interns_num_shifts_array), total_points]

    a lower score is better - think like a loss function.
    :return:
    """
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
                                              best_week_calendars_list, input_statistics, is_last_week):
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
            # todo: go back to step one. do it in create_week_calendars_files.
            # fixme: should only be done if all calendars that were produced using the current permutation are invalid.
            # interns_permutation = return_next_permutation(
            #    interns_permutation, min_num_of_shifts_per_week, num_of_days_in_week, extra_interns_list)
            return False, interns_permutation
    else:  # since there are no extra interns there is no need to try other permutations.
        print("best_week_calendar is none and not extra interns - break. ", "week no. is ", week_number)
        return True, interns_permutation


def handle_current_week_assignment(preferations_df, max_num_of_shifts_per_week, max_num_of_shifts,
                                   min_num_of_shifts_per_week, min_num_of_shifts, calendar_to_fill,
                                   week_number, best_week_calendars_list,
                                   monthly_interns_num_shifts_array, monthly_interns_points_array,
                                   interns_permutation, extra_interns_list, best_week_calendar,
                                   current_week_calendars, input_statistics):
    """
    to create a different useful permutation each time best_week_calendar is None, we calculate all the
    different combinations of interns to be extra (a subset of the power set, of all list of a certain
    length)  - see `else` clause, count no. useful permutations
    """
    should_create_week_calendars_as_numpy_files = False
    # loop will break either if best_week_calendar is not None, or if all useful permutation have been tried.
    while best_week_calendar is None:  # next try hopefully the permutation will be different
        # insert the best(s) current week
        # todo: choose multiple best calendars - NP-hard
        # todo: allow re-calculation of the numpy files with different permutations order
        # goes through all calculated week_calendars of the current week.
        best_week_calendar = \
            return_best_valid_current_week(calendar_to_fill, week_number,
                                           min_num_of_shifts, max_num_of_shifts, best_week_calendars_list,
                                           preferations_df.shape[0], interns_permutation,
                                           monthly_interns_num_shifts_array, monthly_interns_points_array,
                                           current_week_calendars, input_statistics)
        if best_week_calendar is not None:
            calendar_to_fill, best_week_calendars_list, monthly_interns_num_shifts_array, monthly_interns_points_array \
                = update_according_to_valid_best_week_calendar(calendar_to_fill, week_number, best_week_calendar,
                                                               best_week_calendars_list,
                                                               monthly_interns_num_shifts_array,
                                                               monthly_interns_points_array)
        else:
            print("a best calendar in week ", week_number, "was none ")
            stop_search_for_best_week_calendar, interns_permutation = \
                should_stop_search_for_best_week_calendar(calendar_to_fill, week_number, preferations_df,
                                                          min_num_of_shifts, extra_interns_list, interns_permutation,
                                                          min_num_of_shifts_per_week)
            if stop_search_for_best_week_calendar:
                print("stop_search_for_best_week_calendar")
                break
            else:
                should_create_week_calendars_as_numpy_files = True
                # interns_permutation = return_next_permutation(
                #     interns_permutation, min_num_of_shifts_per_week, num_of_days_in_week, extra_interns_list)
                # return_current_week_calendars(calendar_to_fill, week_number, preferations_df, min_num_of_shifts,
                #                               interns_permutation, max_num_of_shifts_per_week,
                #                               min_num_of_shifts_per_week,
                #                               last_day_of_week, first_day_of_week, extra_interns_list,
                #                               current_week_calendars_were_calculated)
                # TODO: CALCULATE AGAIN CURRENT_WEEK_CALENDAR FILES  # todo in step 1
    return (calendar_to_fill, best_week_calendars_list, monthly_interns_num_shifts_array,
            monthly_interns_points_array, interns_permutation, best_week_calendar,
            should_create_week_calendars_as_numpy_files)


def create_full_calendar(preferations_df, max_num_of_shifts_per_week, max_num_of_shifts, min_num_of_shifts_per_week,
                         min_num_of_shifts, number_of_days_in_month, first_day_of_month, interns_permutation):
    """

    extra_interns_list is utilized only if there are extra interns, but since it's in a called subroutine, we will
    compute it anyway each week to avoid a warning.

    use best_week_calendars_list for statistical analysis.

    min_num_of_shifts_per_week == max_num_of_shifts_per_week == 1. if 0 or >1 it takes to long to calculate. do this
    adjustment on the output of this subroutine.
    :return:
    """  # todo: in create_week_calendars files, also save the used partitions, to use here
    # local variables initiation based on the subroutine's input
    calendar_to_fill = create_empty_calendar(number_of_days_in_month, first_day_of_month)
    best_week_calendars_list = []
    # to calculate and np.save or to np.load current_week_calendars
    current_week_calendars_were_calculated = True  # is called after create_week_calendars_as_numpy_files
    # is zeroed only here
    monthly_interns_num_shifts_array = np.zeros(preferations_df.shape[0])
    monthly_interns_points_array = np.zeros(preferations_df.shape[0])
    # to make up for not allowing 0 shifts while there are too many interns, try different interns permutations
    # to make sure that each week different extra interns are last and thus do not do any shifts.
    for week_number in range(calendar_to_fill.shape[0]):
        best_week_calendar = None
        num_of_days_in_week = calculate_num_of_days_in_week(calendar_to_fill, week_number)
        if num_of_days_in_week <= 0:  # the last week of the month might have 0 days
            continue
        extra_interns_list = create_extra_interns_list(interns_permutation, min_num_of_shifts_per_week,
                                                       num_of_days_in_week)
        calendar_to_fill, best_week_calendars_list, monthly_interns_num_shifts_array, monthly_interns_points_array, \
            interns_permutation, best_week_calendar = \
            handle_current_week_assignment(preferations_df, max_num_of_shifts_per_week, max_num_of_shifts,
                                           min_num_of_shifts_per_week, min_num_of_shifts, calendar_to_fill,
                                           week_number, best_week_calendars_list,
                                           monthly_interns_num_shifts_array, monthly_interns_points_array,
                                           interns_permutation, extra_interns_list, best_week_calendar,
                                           current_week_calendars_were_calculated)
    return calendar_to_fill
