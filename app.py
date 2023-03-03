# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np

from fill_calendar_from_chosen_week_till_month_end import fill_calendar_from_chosen_week_till_month_end

# min_num_of_shifts_per_week must be >= 1 to allow good runtimes
# is program was split into 2 parts to use dash in the middle
# todo: # add as input all current_week_calendars, to integrate with dash.

# assign each not-full calendar one shift of each intern at the time - re-check fullness.

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
# variance of score between participants in each possibility for the current week

# # jupyter notebook start
# week_number = int(input("enter week number"))
# current_week_calendars = np.load(file=f"jup_week_calendars{week_number}.npy", mmap_mode=None, allow_pickle=True,
#                                  fix_imports=False, encoding="ASCII")
# current_week_calendars_scores = []
# for calendar in current_week_calendars:
#     calendar = calendar[1]
#     current_week_calendars_scores.append(calendar)
#     # print(calendar)
# current_week_calendars_scores = np.array(current_week_calendars_scores)
# # shape is [no. elements, 2, no. interns]
# print(current_week_calendars_scores.shape)
#
# df = pd.DataFrame(np.transpose(current_week_calendars_scores[:, 0, :], (1, 0)))
# df = df.describe()
# fig = px.bar(df.loc["std"].sort_values().reset_index()["std"])  # fixme: make a label that says no. of holes.
# # jupyter end


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
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value')
)
def foo(week_number=0, input_statistics=None):
    # TODO: add an argument to fill_calendar_from_chosen_week_till_month_end that says whether to calculate or load.
    calendar_to_fill, best_week_calendars_list, extra_interns_lists, monthly_interns_num_shifts_array, \
        monthly_interns_points_array = fill_calendar_from_chosen_week_till_month_end(preferations_df_xlsx_path=
                                                                                     "preferations_df.xlsx",
                                                                                     max_num_of_shifts_per_week=1,
                                                                                     min_num_of_shifts_per_week=1,
                                                                                     number_of_days_in_month=30,
                                                                                     first_day_of_month=1,  # monday
                                                                                     min_num_of_shifts=4,
                                                                                     max_num_of_shifts=6,
                                                                                     week_number=0,
                                                                                     input_statistics=[1])
    current_week_calendars = best_week_calendars_list[week_number]
    # np.load(file=f"jup_week_calendars{week_number}.npy", mmap_mode=None, allow_pickle=True,
    #                              fix_imports=False, encoding="ASCII")
    current_week_calendars_scores = []
    for calendar in current_week_calendars:
        calendar = calendar[1]
        current_week_calendars_scores.append(calendar)
        # print(calendar)
    current_week_calendars_scores = np.array(current_week_calendars_scores)
    # shape is [no. elements, 2, no. interns]
    # print(current_week_calendars_scores.shape)

    current_week_calendars_scores
    # outputs
    df = pd.DataFrame(np.transpose(current_week_calendars_scores[:, 0, :], (1, 0)))
    fig = px.bar(df.loc["std"].sort_values().reset_index()["std"])  # fixme: make a label that says no. of holes.
    return df, fig


app.layout = html.Div(children=[

    html.H1(children='Smart Shift Assignment'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    html.Div(children="week 0"),
    dcc.Graph(
        id='example-graph',
        figure=fig
    ),

    html.Div(children="week 1"),
    dcc.Graph(
        id='example-graph',
        figure=fig
    ),

    html.Div(children="week 2"),
    dcc.Graph(
        id='example-graph',
        figure=fig
    ),

    html.Div(children="week 3"),
    dcc.Graph(
        id='example-graph',
        figure=fig
    ),

    html.Div(children="week 4"),
    dcc.Graph(
        id='example-graph',
        figure=fig
    ),

    generate_table(df),  # fixme: display calendar_to_fill or holes_calendar

    html.H6("Change the value in the text box to see callbacks in action!"),
    html.Div([
        "Input: ",
        dcc.Input(id='my-input', value='initial value', type='text')
    ]),
    html.Br(),
    html.Div(id='my-output'),
])

if __name__ == '__main__':
    app.run_server()  # debug=True)
# add ability to choose week.
