# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np

from fill_calendar_from_chosen_week_till_month_end import fill_calendar_from_chosen_week_till_month_end

# min_num_of_shifts_per_week must be >= 1 to allow good runtimes
# todo: # add as input all current_week_calendars, to integrate with dash.
# assign each not-full calendar one shift of each intern at the time - re-check fullness.

# see https://plotly.com/python/px-arguments/ for more options
# variance of score between participants in each possibility for the current week

app = Dash(__name__)


def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value')
)
def update_output_div(input_value):
    return f'Output: {input_value}'


@app.callback(
    Output(component_id='update_best_calendars_from_current_week-output', component_property='children'),
    Input(component_id='update_best_calendars_from_current_week-input', component_property='value')
)
def update_best_calendars_from_current_week(week_number):
    preferations_df_xlsx_path = "\\Users\\golan\\Downloads\\medical-shifts-assignment\\preferations_df.xlsx"
    max_num_of_shifts_per_week = 1
    min_num_of_shifts_per_week = 1
    number_of_days_in_month = 30
    first_day_of_month = 1  # monday
    min_num_of_shifts = 4
    max_num_of_shifts = 6
    week_number = int(week_number)
    input_statistics = None
    # TODO: add an argument to fill_calendar_from_chosen_week_till_month_end that says whether to calculate or load.
    calendar_to_fill, best_week_calendars_list, extra_interns_lists, monthly_interns_num_shifts_array, \
        monthly_interns_points_array, all_week_calendars_list = \
        fill_calendar_from_chosen_week_till_month_end(preferations_df_xlsx_path,
                                                      max_num_of_shifts_per_week,
                                                      min_num_of_shifts_per_week,
                                                      number_of_days_in_month,
                                                      first_day_of_month,
                                                      min_num_of_shifts,
                                                      max_num_of_shifts,
                                                      week_number,
                                                      input_statistics)
    # return calendar_to_fill, best_week_calendars_list, extra_interns_lists, monthly_interns_num_shifts_array, \
    #     monthly_interns_points_array, all_week_calendars_list
    return week_number


def display_current_week_solutions_statistics(path_to_folder, week_number):
    """
    to choose statistics for update_best_calendars_from_current_week.
    """
    path_to_file = path_to_folder + f"\\week_calendars{week_number}.npy"
    current_week_calendars = np.load(file=path_to_file, mmap_mode=None, allow_pickle=True, fix_imports=False,
                                     encoding="ASCII")
    current_week_calendars_scores = []
    for calendar in current_week_calendars:
        current_week_calendars_scores.append(calendar[1])
        # print(calendar)
    current_week_calendars_scores = np.array(current_week_calendars_scores)
    # shape is [no. elements, 2, no. interns]
    # print(current_week_calendars_scores.shape)

    # outputs
    df = pd.DataFrame(np.transpose(current_week_calendars_scores[:, 0, :], (1, 0)))
    df = df.describe()
    fig = px.bar(df.loc["std"].sort_values().reset_index()["std"])  # fixme: make a label that says no. of holes.
    return df, fig


# df_first_week, fig_first_week = foo(week_number=0, input_statistics=None)
# df_second_week, fig_second_week = foo(week_number=1, input_statistics=None)
# df_third_week, fig_third_week = foo(week_number=2, input_statistics=None)
# df_fourth_week, fig_fourth_week = foo(week_number=3, input_statistics=None)
#
df_fifth_week, fig_fifth_week = display_current_week_solutions_statistics(
    path_to_folder="C:\\Users\\golan\\Downloads\\medical-shifts-assignment", week_number=1)
# df_sixth_week, fig_sixth_week = foo(week_number=5, input_statistics=None)

update_best_calendars_from_current_week(1)


app.layout = html.Div(children=[

    html.H1(children='Smart Shift Assignment'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),
    # TODO: display only one week at the time. change week number using callbacks
    html.Div(children="week 1"),
    dcc.Graph(
        id='example-graph',
        figure=fig_fifth_week
    ),

    # html.Div(children="week 1"),
    # dcc.Graph(
    #     id='example-graph',
    #     figure=fig_second_week
    # ),

    # generate_table(df),  # fixme: display calendar_to_fill or holes_calendar
    html.H6("Change the value in the text box to re-calculate from week 4"),
    html.Div([
        "Input: ",
        dcc.Input(id='update_best_calendars_from_current_week-input', value='3', type='text')
    ]),
    html.Br(),
    html.Div(id='update_best_calendars_from_current_week-output'),


    # html.H6("Change the value in the text box to see callbacks in action!"),
    # html.Div([
    #     "Input: ",
    #     dcc.Input(id='my-input', value='initial value', type='text')
    # ]),
    # html.Br(),
    # html.Div(id='my-output'),
    # todo: loading symbol as long as we calculate solutions.
])

if __name__ == '__main__':
    app.run_server()  # debug=True)
# add ability to choose week.
