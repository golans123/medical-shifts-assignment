import numpy as np
import pandas as pd

import create_week_calendars_files
import choose_best_calendars_from_files


def handle_creation_of_week_calendars(calendar_to_fill, week_number, interns_permutation, amount_of_interns,
                                      min_num_of_shifts_per_week, extra_interns_lists, preferations_df,
                                      last_week_number, max_num_of_shifts_per_week, number_of_days_in_month,
                                      first_day_of_month):
    should_break = False
    should_continue = False
    current_week_calendars = None
    first_day_of_week, last_day_of_week = create_week_calendars_files.calculate_first_and_last_day_of_week(
        calendar_to_fill.iloc[week_number, :])
    num_of_days_in_week = create_week_calendars_files.calculate_num_of_days_in_week(calendar_to_fill, week_number)
    # to make up for not allowing 0 shifts while there are too many interns, try different interns permutations
    # to make sure that each week different extra interns are last and thus do not do any shifts.
    if not (create_week_calendars_files.there_are_extra_interns(
            amount_of_interns, min_num_of_shifts_per_week, num_of_days_in_week) and
            extra_interns_lists[week_number]):
        # fixme: what if there are no extra interns... oh!! then the permutation is just a matter of points
        #  (not success) :)
        extra_interns_lists[week_number] = \
            create_week_calendars_files.create_extra_interns_list(amount_of_interns,
                                                                  min_num_of_shifts_per_week,
                                                                  num_of_days_in_week)
    interns_permutation = create_week_calendars_files.create_initial_interns_permutation(
        number_of_interns=preferations_df.shape[0]) if (not interns_permutation) \
        else create_week_calendars_files.return_next_permutation(interns_permutation, min_num_of_shifts_per_week,
                                                                 num_of_days_in_week,
                                                                 extra_interns_lists[week_number])
    if not interns_permutation:
        if week_number == 0:  # fixme: instead of 0, use variable_initial week, for manual tweaking of a middle week.
            should_break = True
            return should_break, should_continue, current_week_calendars, week_number
        else:
            # to force re-calculation of extra_interns_list of all following weeks.
            for i in range(week_number, last_week_number + 1):
                extra_interns_lists[i] = []
            # try a different useful permutation of the previous week
            week_number -= 1
            should_continue = True
            return should_break, should_continue, current_week_calendars, week_number
    current_week_calendars = \
        create_week_calendars_files.create_week_calendars_as_numpy_file(preferations_df=preferations_df,
                                                                        max_num_of_shifts_per_week=
                                                                        max_num_of_shifts_per_week,
                                                                        min_num_of_shifts_per_week=
                                                                        min_num_of_shifts_per_week,
                                                                        number_of_days_in_month=
                                                                        number_of_days_in_month,
                                                                        first_day_of_month=first_day_of_month,
                                                                        week_number=week_number,
                                                                        first_day_of_week=first_day_of_week,
                                                                        last_day_of_week=last_day_of_week,
                                                                        num_of_days_in_week=num_of_days_in_week,
                                                                        interns_permutation=interns_permutation)
    return should_break, should_continue, current_week_calendars, week_number


def choose_best_calendar(calendar_to_fill, current_week_calendars, best_week_calendars_list,
                         monthly_interns_num_shifts_array, monthly_interns_points_array, week_number, preferations_df,
                         max_num_of_shifts_per_week, max_num_of_shifts,
                         min_num_of_shifts_per_week,
                         min_num_of_shifts, extra_interns_lists, input_statistics):
    should_break = False
    best_week_calendar = None
    should_create_week_calendars_as_numpy_files = True
    num_of_days_in_week = create_week_calendars_files.calculate_num_of_days_in_week(
        calendar_to_fill, week_number)
    interns_permutation = create_week_calendars_files.create_initial_interns_permutation(
        number_of_interns=preferations_df.shape[0])
    if num_of_days_in_week <= 0:  # the last week of the month might have 0 days
        should_break = True
        return should_break, calendar_to_fill, best_week_calendars_list, monthly_interns_num_shifts_array, \
            monthly_interns_points_array, interns_permutation, best_week_calendar, \
            should_create_week_calendars_as_numpy_files
    calendar_to_fill, best_week_calendars_list, monthly_interns_num_shifts_array, monthly_interns_points_array, \
        interns_permutation, best_week_calendar, should_create_week_calendars_as_numpy_files = \
        choose_best_calendars_from_files.handle_current_week_assignment(preferations_df,
                                                                        max_num_of_shifts_per_week,
                                                                        max_num_of_shifts,
                                                                        min_num_of_shifts_per_week,
                                                                        min_num_of_shifts,
                                                                        calendar_to_fill,
                                                                        week_number, best_week_calendars_list,
                                                                        monthly_interns_num_shifts_array,
                                                                        monthly_interns_points_array,
                                                                        interns_permutation,
                                                                        extra_interns_lists[week_number],
                                                                        best_week_calendar,
                                                                        current_week_calendars, input_statistics)
    return should_break, calendar_to_fill, best_week_calendars_list, monthly_interns_num_shifts_array, \
        monthly_interns_points_array, interns_permutation, best_week_calendar, \
        should_create_week_calendars_as_numpy_files


def fill_calendar_from_chosen_week_till_month_end(preferations_df_xlsx_path="preferations_df.xlsx",
                                                  max_num_of_shifts_per_week=1,
                                                  min_num_of_shifts_per_week=1,
                                                  number_of_days_in_month=30,
                                                  first_day_of_month=1,  # monday
                                                  min_num_of_shifts=4,
                                                  max_num_of_shifts=6,
                                                  week_number=0,
                                                  input_statistics=None):
    # todo: add as input all current_week_calendars, to integrate with dash.
    # todo: also handle a case of not having enough interns - still unoccupied shifts after assignment.
    #  use calendar_week_is_full for that purpose
    # todo: allow inserting some shifts / weeks manually.
    # todo: decide upon the structure of `input_statistics`.
    # part 1 - np file
    # part 2 - dash
    # part 3 - choose best week
    preferations_df = create_week_calendars_files.create_preferations_df(preferations_df_xlsx_path)
    preferations_df = preferations_df  # .iloc[0:2]
    preferations_df.loc[6] = pd.Series(preferations_df.iloc[5])
    preferations_df.loc[7] = pd.Series(preferations_df.iloc[5])
    preferations_df.iloc[:, 1:5] = -1  # disabling bans list fixme
    amount_of_interns = preferations_df.shape[0]

    calendar_to_fill = create_week_calendars_files.create_empty_calendar(number_of_days_in_month, first_day_of_month)
    best_week_calendars_list = []
    extra_interns_lists = [[], [], [], [], [], []]  # has an extra_interns_list for each week.
    all_week_calendars_list = [[], [], [], [], [], []]  # todo: for the app
    # are zeroed only here
    monthly_interns_num_shifts_array = np.zeros(amount_of_interns)
    monthly_interns_points_array = np.zeros(amount_of_interns)

    last_week_number = calendar_to_fill.shape[0] - 1
    interns_permutation = None

    while week_number <= last_week_number:
        # part 1 - create_week_calendars_as_numpy_file. the file can be used to choose different threshold without
        #  re-calculating current_week_calendars.
        should_break, should_continue, current_week_calendars, week_number = \
            handle_creation_of_week_calendars(calendar_to_fill, week_number, interns_permutation, amount_of_interns,
                                              min_num_of_shifts_per_week, extra_interns_lists, preferations_df,
                                              last_week_number, max_num_of_shifts_per_week, number_of_days_in_month,
                                              first_day_of_month)
        if should_break:  # no useful permutation in week 0
            break
        if should_continue:  # check a different useful permutation in the current week
            continue
        all_week_calendars_list[week_number].append(current_week_calendars)  # todo: for the app
        # part 2 - choose_best_calendar from file
        should_break, calendar_to_fill, best_week_calendars_list, \
            monthly_interns_num_shifts_array, monthly_interns_points_array, interns_permutation, \
            best_week_calendar, should_create_week_calendars_as_numpy_files = \
            choose_best_calendar(calendar_to_fill, current_week_calendars, best_week_calendars_list,
                                 monthly_interns_num_shifts_array, monthly_interns_points_array, week_number,
                                 preferations_df,
                                 max_num_of_shifts_per_week, max_num_of_shifts,
                                 min_num_of_shifts_per_week,
                                 min_num_of_shifts, extra_interns_lists, input_statistics)

        if should_break:
            break
        if best_week_calendar is None:
            continue  # do re-calculate extra interns list
        else:
            week_number += 1  # calculate extra interns list
    return calendar_to_fill, best_week_calendars_list, extra_interns_lists, monthly_interns_num_shifts_array, \
        monthly_interns_points_array, all_week_calendars_list


if __name__ == '__main__':
    main_calendar_to_fill = fill_calendar_from_chosen_week_till_month_end(
        preferations_df_xlsx_path="preferations_df.xlsx",
        max_num_of_shifts_per_week=1,
        min_num_of_shifts_per_week=1,
        number_of_days_in_month=30,
        first_day_of_month=1,  # monday
        min_num_of_shifts=4,
        max_num_of_shifts=6,
        week_number=0,
        input_statistics=[1])
