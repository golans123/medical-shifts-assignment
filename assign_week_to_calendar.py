import numpy as np
import pandas as pd
import copy

iteration = 0


def create_empty_calendar(number_of_days_in_month, first_day_of_month):
    """
    a dataframe whose elements are lists cannot be deeply copied. hence the use
     of tuples.
     `None`` was used since pd.NA has an ambiguous boolean value.
    :param number_of_days_in_month: int, 28 <= first_day_of_month <= 31
    :param first_day_of_month: int, 0 <= first_day_of_month <= 6
    :return:
    """
    days_of_the_week = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturaday"]
    week_number = np.arange(6).astype(int)
    current_day_of_month = 1
    current_day_of_week = first_day_of_month
    empty_calendar = pd.DataFrame(columns=days_of_the_week, index=week_number)
    for i in week_number:
        appended_days = [[j, None] for j in range(current_day_of_week, 7)]
        empty_calendar.iloc[i, current_day_of_week:7] = appended_days
        current_day_of_week = 0
        current_day_of_month = current_day_of_month + len(appended_days)
    # emptying days will always happen at the last week
    if current_day_of_month > number_of_days_in_month:
        number_of_days_to_delete = current_day_of_month - number_of_days_in_month - 1
        if number_of_days_to_delete <= 7:
            empty_calendar.iloc[-1, 7 - number_of_days_to_delete:7] = \
                [None for j in range(number_of_days_to_delete)]
        else:
            empty_calendar.iloc[-1] = [None for j in range(7)]
            empty_calendar.iloc[-2, 14 - number_of_days_to_delete:7] = \
                [None for j in range(number_of_days_to_delete - 7)]
    return empty_calendar


def update_date(current_day_in_week, current_week_number, day_in_month, last_day_in_week, skip_tomorrow=False):
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
                                     current_day_in_week, intern_index, preferations_df, day_in_month):
    # input_clanedar[1][0], input_clanedar[1][0], which are lists, should be deeply copied too
    calendar_with_shift_today = [input_calendar[0].copy(deep=True),
                                 [copy.deepcopy(input_calendar[1][0]), copy.deepcopy(input_calendar[1][1])],
                                 input_calendar[2]]
    calendar_with_shift_today[0].iloc[current_week_number].iloc[current_day_in_week]\
        = [current_day_in_week, intern_index]
    # update points
    calendar_with_shift_today[1][0][intern_index] += preferations_df.iloc[intern_index][5 + day_in_month]
    calendar_with_shift_today[1][1][intern_index] += 1
    calendar_with_shift_today[2] += preferations_df.iloc[intern_index][5 + day_in_month]
    return calendar_with_shift_today


def update_calendars(input_calendar, intern_index, preferations_df, day_in_month,
                     shift_number, shifts_to_do, current_week_number, current_day_in_week,
                     last_day_in_week, first_day_of_month, number_of_days_in_month,
                     calendars):
    original_current_day_in_week, original_current_week_number, original_day_in_month = \
        current_day_in_week, current_week_number, day_in_month
    # todo: day +=2, can't do two shifts in a row
    # with shift today, check if day is not already occupied
    if input_calendar[0].iloc[current_week_number].iloc[current_day_in_week][1] is None:
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
    calendar_without_shift_today = [input_calendar[0].copy(deep=True),
                                    [input_calendar[1][0], input_calendar[1][1]], input_calendar[2]]
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
    there is no need to return calendars in the recursion since it's immutable
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
        calendars.append(input_calendar)
        return calendars
    else:
        # not in bans list
        if day_in_month not in preferations_df.iloc[intern_index, 1:5].tolist():
            # recursion stop condition - end of month (day_in_month >= 0)
            # todo: == vs >=. day in month starts at zero, skip tomorrow.
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
            # fixme: for the case where the last day is banned, checking arrival for the last day
            #  would not suffice, so we need... complete. also, if input_calendar is already full
            if ((day_in_month-1) == number_of_days_in_month) or (day_in_month == number_of_days_in_month):
                return calendars
            else:  # fixme
                # without shift today
                current_day_in_week, current_week_number, day_in_month = \
                    update_date(current_day_in_week, current_week_number, day_in_month, last_day_in_week, skip_tomorrow=False)
                calendar_without_shift_today = [input_calendar[0].copy(deep=True),
                                                [input_calendar[1][0], input_calendar[1][1]], input_calendar[2]]
                return assign_intern_recursively(shift_number, shifts_to_do, current_week_number,
                                                 current_day_in_week, last_day_in_week,
                                                 preferations_df, calendar_without_shift_today, intern_index,
                                                 first_day_of_month, day_in_month,
                                                 number_of_days_in_month, calendars)


def assign_intern_to_all_possible_combinations(num_of_shifts,
                                               calendars, last_day_in_week, preferations_df,
                                               intern_index, first_day_of_month, number_of_days_in_month):
    # reset calendars before each intern, since a calendar without all
    # interns should not be saved to `calendars`
    new_calendars = []
    # min_points = np.inf
    # try all possible amounts of shifts
    for i in range(num_of_shifts):
        for j in range(len(calendars)):
            new_calendars = assign_intern_recursively(shift_number=0, shifts_to_do=i, current_week_number=0,
                                                      current_day_in_week=0,
                                                      last_day_in_week=last_day_in_week,
                                                      preferations_df=preferations_df,
                                                      input_calendar=calendars[j],
                                                      intern_index=intern_index,
                                                      first_day_of_month=first_day_of_month,
                                                      day_in_month=0,
                                                      number_of_days_in_month=number_of_days_in_month,
                                                      calendars=new_calendars)
            # fixme: this test line might need to catch more cases
            # for the case where all calendars are full
            if not new_calendars:
                new_calendars = calendars
    # assign to calendars the list of all most up-to-date calendars
    return new_calendars


def return_empty_weekly_calendar(first_day_of_week, last_day_of_week):
    days_of_the_week = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturaday"]
    days_of_the_week = days_of_the_week[first_day_of_week:last_day_of_week+1]
    empty_calendar = pd.DataFrame(columns=days_of_the_week, index=[0])
    appended_days = [[j, None] for j in range(first_day_of_week, last_day_of_week + 1)]
    empty_calendar.iloc[0, first_day_of_week:last_day_of_week + 1] = \
        appended_days[first_day_of_week:last_day_of_week + 1]
    return empty_calendar


def return_valid_weekly_calendars(preferations_df, max_num_of_shifts, min_num_of_shifts,
                                  number_of_days_in_month, first_day_of_month, first_day_in_week,
                                  last_day_in_week):
    """
    each cell of a calendar is a tuple (intern_name, day_num).

    algorithm idea:
    * using recursion according to the chosen number of shifts.
    * no two shifts in a row.

    the theoretical amount of possible calendars in picking any combinations of
    interns, with returns

     the order of the recursion (which intern is first, second, etc.) is
     important as interns might do a different amount of shifts (some 4,
     some 5, some 6...). so instead of trying all orders, we try one order and
     all possible amounts of shifts for each intern.

     for later development:
     * ER shifts are more expensive than department shifts.
     * consider adding shift weight/points to the calendar-cell tuple in order
     to choose best calendar.

    :param first_day_in_week:
    :param last_day_in_week:
    :param min_num_of_shifts:
    :param preferations_df:
    :param max_num_of_shifts:
    :param number_of_days_in_month:
    :param first_day_of_month:
    :return:
    """
    # (calendar, (interns_points_list, interns_num_shifts_list), total_points)
    week_calendars = [[return_empty_weekly_calendar(first_day_in_week, last_day_in_week),
                      [np.zeros(preferations_df.shape[0]), np.zeros(preferations_df.shape[0])], 0]]
    not_full_calendars = week_calendars
    while not_full_calendars:
        # for each calendar, while week is not full, for each intern, assign one shift per intern.
        for i in range(0, preferations_df.shape[0]):
            week_calendars = assign_intern_to_all_possible_combinations(num_of_shifts=1,
                                                                        calendars=week_calendars,
                                                                        last_day_in_week=last_day_in_week,
                                                                        preferations_df=preferations_df, intern_index=i,
                                                                        first_day_of_month=first_day_of_month,
                                                                        number_of_days_in_month=number_of_days_in_month)
        not_full_calendars = []  # fixme - just to test
    # assign each not-full calendar one shift of each intern at the time
    # not_full_calendars = filter_not_full_calendars(calendars)
    # use the not-full calendars
    # full_calendars = [].append(filter_full_calendars(calendars, not_full_calendars))
    # for i in range(max_num_of_shifts-min_num_of_shifts):
    # assign each not-full calendar one shift of each intern at the time - re-check fullness.
    return week_calendars
