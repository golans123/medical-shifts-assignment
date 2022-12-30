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


if __name__ == '__main__':
    main_preferations_df = create_preferations_df("preferations_df.xlsx")
    # fixme: there is an error if not all can do minimal shifts
    main_preferations_df = main_preferations_df.iloc[0:3]
    first_day_of_week = 1
    last_day_of_week = 3
    main_calendars = assign_week_to_calendar.return_valid_weekly_calendars(
        preferations_df=main_preferations_df,
        max_num_of_shifts=2,
        min_num_of_shifts=1,
        number_of_days_in_month=5,
        first_day_of_month=0,
        first_day_in_week=1,
        last_day_in_week=4)

    # print best 10 for each category:
    # total points, uniform points dist., uniform num. shifts dist.
    # lowest_points_calendars = np.array(main_calendars[2]).sort(axis=0)[:10]
    # print(np.array(main_calendars)[:][2])
    # lowest_points_calendars_indices = np.where(main_calendars == lowest_points_calendars)
    # lowest_points_calendars = main_calendars[lowest_points_calendars_indices]
    # for main_i in range(len(main_calendars)):
    #     print(lowest_points_calendars[main_i][0])
    #     print()
    #     print(lowest_points_calendars[main_i][1][0])
    #     print(lowest_points_calendars[main_i][1][1])
    #     print(lowest_points_calendars[main_i][2])
    #     print("\n")
    for main_i in range(len(main_calendars)):
        if calendar_is_full(main_calendars[main_i][0]):
            print(main_calendars[main_i][0])
            print()
            print(main_calendars[main_i][1][0])
            print(main_calendars[main_i][1][1])
            print(main_calendars[main_i][2])
            print("\n")