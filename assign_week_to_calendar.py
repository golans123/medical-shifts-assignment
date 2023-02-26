import numpy as np
import pandas as pd
import copy

iteration = 0


def return_empty_weekly_calendar(first_day_of_week, last_day_of_week):
    days_of_the_week = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturaday"]
    days_of_the_week = days_of_the_week[first_day_of_week:last_day_of_week + 1]
    empty_calendar = pd.DataFrame(columns=days_of_the_week, index=[0])
    # we use np.nan end not None since one can't use np.isnan on a None value
    appended_days = [[j, np.nan] for j in range(first_day_of_week, last_day_of_week + 1)]
    empty_calendar[0, :] = appended_days  # empty_calendar.iloc[0, :] = appended_days
    return empty_calendar


def update_date(current_day_in_week, current_week_number, day_in_month, last_day_in_week, skip_tomorrow=False):
    """
    an intern can't do two shift two-days in a row.
    we always send only one week in the current use of the program
    :param current_day_in_week:
    :param current_week_number:
    :param day_in_month:
    :param last_day_in_week:
    :param skip_tomorrow:
    :return:
    """
    if not skip_tomorrow:
        day_in_month += 1
        if current_day_in_week == last_day_in_week:  # last_day_in_week = 6
            current_day_in_week = 0
            current_week_number += 1
        else:
            current_day_in_week += 1
    else:
        day_in_month += 2
        if current_day_in_week == last_day_in_week:  # last_day_in_week = 6
            current_day_in_week = 1
            current_week_number += 1
        elif current_day_in_week == (last_day_in_week - 1):
            current_day_in_week = 0
            current_week_number += 1
        else:
            current_day_in_week += 2
    return current_day_in_week, current_week_number, day_in_month


def create_calendar_with_shift_today(input_calendar, current_week_number,
                                     current_day_in_week, intern_index,
                                     preferations_df, day_in_month):
    # input_clanedar[1][0] and input_clanedar[1][0], which are lists, should be
    #  deeply copied too
    # input_calendar[0].copy(deep=True),
    calendar_with_shift_today = [copy.deepcopy(input_calendar[0]),
                                 [copy.deepcopy(input_calendar[1][0]), copy.deepcopy(input_calendar[1][1])],
                                 input_calendar[2]]
    # calendar_with_shift_today[0].iloc[current_week_number].iloc[current_day_in_week]
    calendar_with_shift_today[0][current_week_number][current_day_in_week] = [current_day_in_week, intern_index]
    # update points
    calendar_with_shift_today[1][0][intern_index] += preferations_df.iloc[intern_index][5 + day_in_month]
    calendar_with_shift_today[1][1][intern_index] += 1
    calendar_with_shift_today[2] += preferations_df.iloc[intern_index][5 + day_in_month]
    return calendar_with_shift_today


def update_calendars(input_calendar, intern_index, preferations_df, day_in_month,
                     shift_number, shifts_to_do, current_week_number, current_day_in_week,
                     last_day_in_week, first_day_of_month, number_of_days_in_month,
                     calendars):
    """
    can't do two shifts in a row
    :param input_calendar:
    :param intern_index:
    :param preferations_df:
    :param day_in_month:
    :param shift_number:
    :param shifts_to_do:
    :param current_week_number:
    :param current_day_in_week:
    :param last_day_in_week:
    :param first_day_of_month:
    :param number_of_days_in_month:
    :param calendars:
    :return:
    """
    original_current_day_in_week, original_current_week_number, original_day_in_month = \
        current_day_in_week, current_week_number, day_in_month
    # with shift today, check if day is not already occupied
    # input_calendar[0].iloc[current_week_number].iloc[current_day_in_week][1] is None:
    if input_calendar[0][current_week_number][current_day_in_week][1] is np.nan:
        calendar_with_shift_today = \
            create_calendar_with_shift_today(input_calendar,
                                             current_week_number,
                                             current_day_in_week,
                                             intern_index,
                                             preferations_df, day_in_month)
        current_day_in_week, current_week_number, day_in_month = \
            update_date(current_day_in_week, current_week_number, day_in_month, last_day_in_week, skip_tomorrow=True)

        calendars = assign_intern_recursively(shift_number + 1, shifts_to_do, current_week_number,
                                              current_day_in_week,
                                              last_day_in_week,
                                              preferations_df, calendar_with_shift_today, intern_index,
                                              first_day_of_month, day_in_month,
                                              number_of_days_in_month, calendars)
    # to avoid updating the date twice + different skip_tomorrow argument
    current_day_in_week, current_week_number, day_in_month = \
        update_date(original_current_day_in_week, original_current_week_number, original_day_in_month, last_day_in_week,
                    skip_tomorrow=False)
    # without shift today
    # input_calendar[0].copy(deep=True),
    calendar_without_shift_today = [copy.deepcopy(input_calendar[0]),
                                    [copy.deepcopy(input_calendar[1][0]), copy.deepcopy(input_calendar[1][1])],
                                    input_calendar[2]]
    calendars = assign_intern_recursively(shift_number, shifts_to_do, current_week_number,
                                          current_day_in_week,
                                          last_day_in_week,
                                          preferations_df, calendar_without_shift_today, intern_index,
                                          first_day_of_month, day_in_month,
                                          number_of_days_in_month, calendars)
    return calendars


def assign_intern_recursively(shift_number, shifts_to_do, current_week_number,
                              current_day_in_week,
                              last_day_in_week,
                              preferations_df, input_calendar, intern_index,
                              first_day_of_month, day_in_month,
                              number_of_days_in_month, calendars):
    """
    there is no need to return calendars in the recursion since it's
    current_week_number always equals zero since we always send only one week.
    :param shift_number:
    :param shifts_to_do:
    :param current_week_number:
    :param current_day_in_week:
    :param last_day_in_week:
    :param preferations_df:
    :param input_calendar: (calendar, interns_points_list, total_points)
    :param intern_index:
    :param first_day_of_month:
    :param day_in_month:
    :param number_of_days_in_month:
    :param calendars: (calendar, interns_points_list, total_points)
    :return:
    """
    ###############
    if globals()["iteration"] % 1000 == 0:
        print(globals()["iteration"])
    globals()["iteration"] += 1
    ###############
    if shift_number == shifts_to_do:
        if len(calendars) == 0:
            calendars = np.array([input_calendar[0], input_calendar[1], input_calendar[2]], dtype=object)
        else:
            calendars = np.vstack((calendars,
                                   np.array([input_calendar[0], input_calendar[1], input_calendar[2]], dtype=object)))
        return calendars
    else:
        # not in bans list
        if day_in_month not in preferations_df.iloc[intern_index, 1:5].tolist():
            # recursion stop condition - end of month (day_in_month >= 0)
            # we use >= since a day of a month starts at zero, and
            # we also sometimes skip consecutive days
            if day_in_month >= number_of_days_in_month:
                return calendars
            else:  # continue recursion with and without a shift today
                # update input calendar, create calendars to pass to recursion
                calendars = \
                    update_calendars(input_calendar, intern_index,
                                     preferations_df, day_in_month, shift_number,
                                     shifts_to_do, current_week_number, current_day_in_week,
                                     last_day_in_week, first_day_of_month,
                                     number_of_days_in_month, calendars)
                return calendars
        else:
            # recursion stop condition - end of month (day_in_month >= 0)
            # fixme: for the case where the last day is banned, checking equality to the the last day
            #  would not suffice, so we need... complete. also, if input_calendar is already full
            if ((day_in_month - 1) == number_of_days_in_month) or (day_in_month == number_of_days_in_month):
                return calendars
            else:
                # without shift today
                current_day_in_week, current_week_number, day_in_month = \
                    update_date(current_day_in_week, current_week_number, day_in_month, last_day_in_week,
                                skip_tomorrow=False)
                # input_calendar[0].copy(deep=True),
                calendar_without_shift_today = [copy.deepcopy(input_calendar[0]),
                                                [input_calendar[1][0], input_calendar[1][1]], input_calendar[2]]
                return assign_intern_recursively(shift_number, shifts_to_do, current_week_number,
                                                 current_day_in_week, last_day_in_week,
                                                 preferations_df, calendar_without_shift_today, intern_index,
                                                 first_day_of_month, day_in_month,
                                                 number_of_days_in_month, calendars)


def assign_intern_to_all_possible_combinations(min_num_of_shifts, max_num_of_shifts,
                                               calendars, last_day_in_week, preferations_df,
                                               intern_index, first_day_of_month, number_of_days_in_month):
    """
    current_week_number should always be equal to zero since we always send only one week.
    :param min_num_of_shifts:
    :param max_num_of_shifts:
    :param calendars:
    :param last_day_in_week:
    :param preferations_df:
    :param intern_index:
    :param first_day_of_month: is actually `first_day_of_week` since in the current use of the program a month is a
    single week
    :param number_of_days_in_month:
    :return:
    """
    new_calendars = np.array([])
    # try all possible amounts of shifts
    for i in range(min_num_of_shifts, max_num_of_shifts + 1):
        for j in range(len(calendars)):
            new_calendars = assign_intern_recursively(shift_number=0, shifts_to_do=i, current_week_number=0,
                                                      current_day_in_week=first_day_of_month,
                                                      last_day_in_week=last_day_in_week,
                                                      preferations_df=preferations_df,
                                                      input_calendar=calendars[j],
                                                      intern_index=intern_index,
                                                      first_day_of_month=first_day_of_month,
                                                      day_in_month=0,
                                                      number_of_days_in_month=number_of_days_in_month,
                                                      calendars=new_calendars)
            # for the case where all calendars are full
            if not new_calendars.tolist():
                new_calendars = calendars
    # assign to calendars the list of all most up-to-date calendars
    return new_calendars


def return_valid_weekly_calendars(preferations_df, interns_permutation, max_num_of_shifts, min_num_of_shifts,
                                  number_of_days_in_month, current_week, first_day_of_month,
                                  last_day_in_week):
    """
    each cell of a calendar is a tuple (intern_name, day_num).

    algorithm idea:
    * assign the given participants' indices in all possible combinations.
    * also assign scores

    the theoretical amount of possible calendars in picking any combinations of
    interns, with returns

     the order of the recursion (which intern is first, second, etc.) is
     important as interns might do a different amount of shifts (some 4,
     some 5, some 6...). so instead of trying all orders, we try one order and
     all possible amounts of shifts for each intern.
     
    for each calendar, while week is not full, for each intern,
    assign at most one shift per intern.
    reset calendars before each intern, since a calendar without all
    interns should not be saved to `calendars`. this reset is done by assigning
    the value of `assign_intern_to_all_possible_combinations` into `week_calendars.
     
     for later development:
     * ER shifts are more expensive than department shifts.

    :param interns_permutation: an order of the interns' indices (according to preferations df).
    :param current_week: anumpy array of shape (1, 7), each element is [day_number, intern_index], can we used with
    Numba.
    :param last_day_in_week:
    :param min_num_of_shifts:
    :param preferations_df:
    :param max_num_of_shifts:
    :param number_of_days_in_month:
    :param first_day_of_month:
    :return:
    """
    # (calendar, (interns_points_list, interns_num_shifts_list), total_points)
    # todo: current week was return_empty_weekly_calendar(first_day_in_week, last_day_in_week)
    # current week is a numpy  todo: use numba in functions where needed (do benchmarks)
    week_calendars = np.array(
        [[current_week, [np.zeros(preferations_df.shape[0]), np.zeros(preferations_df.shape[0])], 0]], dtype=object)
    # return week_calendars
    for i in interns_permutation:
        week_calendars = assign_intern_to_all_possible_combinations(min_num_of_shifts=min_num_of_shifts,
                                                                    max_num_of_shifts=max_num_of_shifts,
                                                                    calendars=week_calendars,
                                                                    last_day_in_week=last_day_in_week,
                                                                    preferations_df=preferations_df, intern_index=i,
                                                                    first_day_of_month=first_day_of_month,
                                                                    number_of_days_in_month=number_of_days_in_month)
    return week_calendars
