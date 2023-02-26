import numpy as np
import pandas as pd
import plotly.express as px
from random import shuffle
import random
import copy


def historgram(scores_df):
    # scores_df.plot(kind="hist", y=0, bins=50)
    fig = px.histogram(scores_df)
    # fig.show()
    return fig


# if __name__ == '__main__':
