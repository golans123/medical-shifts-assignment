import numpy as np
import pandas as pd

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
    preferations_df["number_of_shifts"] = np.zeros(preferations_df.shape[0])  # todo
    preferations_df["total_points"] = np.zeros(preferations_df.shape[0])
    normalize_points(preferations_df)
    return preferations_df


def create_empty_calendar_test(number_of_days_in_month=5, first_day_of_month=0):

    days_of_the_week = ["sunday", "monday", "tuesday", "wednesday", "thursday"]  # "friday", "saturaday"]
    week_number = np.arange(1).astype(int)
    current_day_of_month = 0
    current_day_of_week = first_day_of_month
    empty_calendar = pd.DataFrame(columns=days_of_the_week, index=week_number)
    for i in week_number:
        appended_days = [[j, None] for j in range(current_day_of_week, number_of_days_in_month)]
        empty_calendar.iloc[i, current_day_of_week:number_of_days_in_month] = appended_days
        current_day_of_week = 0
        current_day_of_month = current_day_of_month + len(appended_days)
    return empty_calendar


# we use a while-loop instead of a for-loop because the number of
#  iterations depends also on the bans lists, so it's hard to calculate
def calendar_is_full(calendar):
    for i in range(calendar.shape[0]):
        for j in range(calendar.shape[1]):
            if calendar.iloc[i, j][1] is None:
                return False
    return True


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


def calculate_first_and_last_day_of_week(next_week_calendar):
    first_day_of_week = 0
    last_day_of_week = 6
    while (first_day_of_week < 6) and (next_week_calendar[first_day_of_week] is np.NAN):
        first_day_of_week += 1
    while (last_day_of_week > 0) and (next_week_calendar[last_day_of_week] is np.NAN):
        last_day_of_week -= 1
    return first_day_of_week, last_day_of_week


def return_best_current_week(empty_calendar, current_week_calendars):
    """
    calculate expectation and variance of `current_week_calendars`
    plot these statistics' distributions.

    check total max num of shifts is not exceeded.

    :param empty_calendar:
    :param current_week_calendars:
    :return:
    """
    best_week_calendar = None
    for i in range(len(current_week_calendars)):  # choose only best calendar(s)
        if calendar_is_full(main_calendars[main_i][0]):
            pass
    return best_week_calendar


def create_full_calendar(preferations_df, max_num_of_shifts, min_num_of_shifts,
                         number_of_days_in_month, first_day_of_month):
    empty_calendar = create_empty_calendar(
        number_of_days_in_month=number_of_days_in_month,
        first_day_of_month=first_day_of_month)
    calendars_to_fill = [empty_calendar]
    first_day_of_week, last_day_of_week = \
        calculate_first_and_last_day_of_week(empty_calendar.iloc[0, :])
    print(first_day_of_week)
    
    # for week_number in range(empty_calendar.shape[0]):
        # current_week_calendars = assign_week_to_calendar.return_valid_weekly_calendars(
        #     preferations_df=main_preferations_df,
        #     max_num_of_shifts=2,
        #     min_num_of_shifts=0,
        #     number_of_days_in_month=last_day_of_week + 1 - first_day_of_week,
        #     current_week=pd.DataFrame(empty_calendar.iloc[week_number, :]),
        #     first_day_of_month=first_day_of_week,
        #     last_day_in_week=last_day_of_week)
        # # for each possible calendar to incorporate the current week
        # for calendar_to_fill in calendars_to_fill:
        #     # insert the best(s) current week todo: check total max num of shifts is not exceeded.
        #     best_week_calendar = return_best_current_week(empty_calendar, current_week_calendars)
        #     if best_week_calendar:
        #         empty_calendar.iloc[week_number, :] = best_week_calendar
        # if week_number < empty_calendar.shape[0] - 1:
        #     first_day_of_week = \
        #         calculate_first_and_last_day_of_week(empty_calendar.iloc[week_number + 1, :])
    return calendars_to_fill


if __name__ == '__main__':
    main_preferations_df = create_preferations_df("preferations_df.xlsx")
    main_preferations_df = main_preferations_df.iloc[0:2]
    main_first_day_of_month = 1
    main_last_day_of_week = 2
    main_preferations_df.iloc[:, 1:5] = -1  # disabling bans list

    main_calendars = create_full_calendar(preferations_df=main_preferations_df,
                                          max_num_of_shifts=2,
                                          min_num_of_shifts=0,
                                          number_of_days_in_month=30,
                                          first_day_of_month=main_first_day_of_month)
    for main_i in range(len(main_calendars)):
        if calendar_is_full(main_calendars[main_i][0]):
            print(main_calendars[main_i][0])
            print()
            # print(main_calendars[main_i][1][0])
            # print(main_calendars[main_i][1][1])
            # print(main_calendars[main_i][2])
            # print("\n")
    print(create_empty_calendar(number_of_days_in_month=30, first_day_of_month=0))

    # print best 10 for each category:
    # total points, uniform points dist., uniform num. shifts dist.
    # lowest_points_calendars = np.array(main_calendars[2]).sort(axis=0)[:10]
    # print(np.array(main_calendars)[:][2])
    # lowest_points_calendars_indices = np.where(main_calendars == lowest_points_calendars)
    # lowest_points_calendars = main_calendars[lowest_points_calendars_indices]

    # assign each not-full calendar one shift of each intern at the time
    # not_full_calendars = filter_not_full_calendars(calendars)
    # use the not-full calendars
    # full_calendars = [].append(filter_full_calendars(calendars, not_full_calendars))
    # for i in range(max_num_of_shifts-min_num_of_shifts):
    # assign each not-full calendar one shift of each intern at the time - re-check fullness.

