# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
import numpy as np

app = Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
# variance of score between participants in each possibility for the current week

week_number = int(input("enter week number"))
current_week_calendars = np.load(file=f"jup_week_calendars{week_number}.npy", mmap_mode=None, allow_pickle=True,
                                 fix_imports=False, encoding="ASCII")
current_week_calendars_scores = []
for calendar in current_week_calendars:
    calendar = calendar[1]
    current_week_calendars_scores.append(calendar)
    # print(calendar)
current_week_calendars_scores = np.array(current_week_calendars_scores)
# shape is [no. elements, 2, no. interns]
print(current_week_calendars_scores.shape)

df = pd.DataFrame(np.transpose(current_week_calendars_scores[:, 0, :], (1, 0)))
df = df.describe()
fig = px.bar(df.loc["std"].sort_values().reset_index()["std"])  # fixme: make a label that says no. of holes.


df = pd.read_excel("preferations_df.xlsx")


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


app = Dash(__name__)


app.layout = html.Div(children=[
    html.H1(children='Smart Shift Assignment'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    ),

    generate_table(df)  # fixme: display calendar_to_fill or holes_calendar
])

if __name__ == '__main__':
    app.run_server(debug=True)
# add ability to choose week.
