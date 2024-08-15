import pprint
import pandas as pd
import psycopg2
from psycopg2 import sql
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.dates as mdates
import plotly.graph_objects as go
import plotly.express as px
import psycopg2
from multiprocessing import Pool
from pandas import DataFrame
import get_data

import warnings
from datetime import datetime
import math
from dash import Dash, dcc, html, Input, Output, callback, ctx, Patch
import dbDetails


athleteLines = {}
athleteFigs = {}
groupLine = None


def seconds_to_minutes_and_seconds(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    if minutes > 60:
        hours = minutes // 60
        minutes -= hours * 60
        return f"{hours}:{minutes}:{remaining_seconds:.0f}"
    else:
        return f"{minutes}:{remaining_seconds:.2f}"


def idToName(inputArray):
    filtered_df = athleteInfo[athleteInfo.index.isin(inputArray)]

    newArray = filtered_df['full_name'].tolist()
    return newArray


def NameToID(inputArray):
    if inputArray is None:
        return None
    newArray = []
    for input_name in inputArray:
        matched_rows = athleteInfo[athleteInfo['full_name'] == input_name]
        if not matched_rows.empty:
            newArray.extend(matched_rows.index.tolist())

    return newArray if newArray else None


# Initialize connection pool


def calculateGroupBestFit(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=106, test_size=0.33, shuffle=True)

    best_degree = 0
    best_r2 = 0
    print("SIZE OF ARRAY {}".format(len(x)))

    for degree in range(1, 7):
        # Perform polynomial regression
        fitting = np.polyfit(x_train, y_train, degree)
        model = np.poly1d(fitting)
        y_testing = model(x_test)

        r2all = r2_score(y_test, y_testing)
        print("R-squared score of test data for degree {} with outliers: {:.4f}".format(degree, r2all))

        y_pred = model(x)
        residuals = y - y_pred

        # Compute the standard deviation of the residuals
        std_dev = np.std(residuals)

        # A common criterion: outliers are points where the residual is more than 2 standard deviations away
        threshold = 1.5 * std_dev
        outlier_indices = np.where(np.abs(residuals) > threshold)[0]

        x_filtered = np.delete(x, outlier_indices)
        y_filtered = np.delete(y, outlier_indices)

        print("REMOVED {} outliers".format(len(x) - len(x_filtered)))

        x_train2, x_test2, y_train2, y_test2 = train_test_split(x_filtered, y_filtered, random_state=107,
                                                                test_size=0.33, shuffle=True)

        mymodel2 = np.polynomial.Polynomial.fit(x_train2, y_train2, degree)

        y_pred2 = mymodel2(x_test2)
        r2inliers = r2_score(y_test2, y_pred2)

        print(
            "R-squared score of test data for degree {} without outliers: {:.4f}".format(degree, r2inliers))

        if r2inliers > best_r2:
            best_r2 = r2inliers
            best_degree = degree
            bestLine = mymodel2

    print("BEST DEGREE IS: {}".format(best_degree))

    return bestLine


def create_GroupLine():
    start = datetime.now()
    x = []
    y = []

    for athlete in listOfAthletes:
        x.extend(listOfDFs[athlete]['age'].to_numpy())
        y.extend(listOfDFs[athlete]['wa_points'].to_numpy())

    poly_function = calculateGroupBestFit(x, y)
    print("FINISHED MAKING GROUPLINE {}".format(datetime.now() - start))

    return poly_function


def concurrentIndividual():
    start = datetime.now()
    listofparam = [[id, listOfDFs[id], groupLine] for id in listOfAthletes]

    nofitCount = 0
    poorfitCount = 0
    notlongCount = 0
    goodfitCount = 0

    with Pool(processes=8) as pool:
        results = pool.starmap(calcIndivConcurrent, listofparam)

    for result in results:
        athlete_id, fig, line, r2 = result
        athleteFigs[athlete_id] = fig
        athleteLines[athlete_id] = line

        if r2 == 0:
            nofitCount += 1
        elif r2 == 10:
            notlongCount += 1
        elif r2 > 0.5:
            goodfitCount += 1
        else:
            poorfitCount += 1

    print("TIME TO RUN CALC INDIVIDUALS: {}".format(datetime.now() - start))
    print("GOOD FITS: {}".format(goodfitCount))
    print("POOR FITS: {}".format(poorfitCount))
    print("NO FITS: {}".format(nofitCount))
    print("NOT BIG ENOUGH: {}".format(notlongCount))
    print("FINISHED CONCURRENT INDIVIDUALS IN {}".format(datetime.now() - start))


def calcIndivConcurrent(athlete_id, athleteDF, groupLine):
    def find_closest_performances_conc(input_ages, x, y, num_closest):
        input_ages = np.asarray(input_ages)
        differences = np.abs(x[:, np.newaxis] - input_ages)

        closest_indices = np.argpartition(differences, num_closest, axis=0)[:num_closest, :]
        # weights = calc_weight(x, y)

        closest_performances = y[closest_indices]
        closest_difference = differences[closest_indices, np.arange(input_ages.size)]
        closest_weights = calc_weight_conc(x[closest_indices], y[closest_indices])

        normalised_difference = closest_difference - np.min(closest_difference) / np.max(closest_difference) - np.min(
            closest_difference)

        # Vectorized weight calculations
        exp_neg_diff = np.exp(-(2 * normalised_difference))

        # Compute the weighted mean for each input age
        mean = np.sum(exp_neg_diff * closest_weights * closest_performances, axis=0) / (
            np.sum(exp_neg_diff * closest_weights, axis=0))  # Avoid division by zero

        return mean

    def calc_weight_conc(x_in, y_in):
        if len(x_in) > 0:
            w = groupLine(x_in) - y_in

            if w.size > 0:  # Ensure w is not empty
                w_min, w_max = np.min(w), np.max(w)
                if w_min != w_max:  # Ensure w has more than one unique value to avoid division by zero
                    w = 3 * np.exp(5 * - ((w - w_min) / (w_max - w_min)))
                else:
                    w = np.ones_like(w)  # Handle case where all elements are the same
            else:
                w = np.array([])  # Handle the case where w is empty

            # w = np.exp(2 * (1 - (w - min_val) / (max_val - min_val)) - 2)
            # w[0] = min(1, 2 * w[0])
            # w[-1] = min(1, 2 * w[0])
            return w
        else:
            print("EMPTY ARRAY CALLED")

    best_r2 = 0
    # athleteDF = listOfDFs[athlete_id]
    fig = go.Figure()
    athleteDF['readablePerformance'] = athleteDF['performance_time'].apply(seconds_to_minutes_and_seconds)

    fig = px.scatter(
        athleteDF, x='age', y='wa_points', hover_data=['event', 'readablePerformance']
    )

    x_athlete = athleteDF['age'].to_numpy()
    y_athlete = athleteDF['wa_points'].to_numpy()
    yearsRunning = int((max(x_athlete) - min(x_athlete)))

    if len(x_athlete) > 8 and yearsRunning > 0:

        x_smooth = np.linspace(min(x_athlete), max(x_athlete), yearsRunning * 24)

        y_smooth = find_closest_performances_conc(x_smooth, x_athlete, y_athlete, 6)
        fig.add_trace(
            go.Scatter(x=x_smooth, y=y_smooth, marker=dict(color='orange'), name="ROLLING AVERAGE"))

        fig.add_trace(go.Scatter(x=x_smooth, y=groupLine(x_smooth), marker=dict(color='green'), name="GROUP LINE"))
        # print(athlete_id)

        x_train, x_test, y_train, y_test = train_test_split(x_smooth, y_smooth, random_state=101)

        for deg in range(1, max(9, yearsRunning)):
            model = np.polynomial.Polynomial.fit(x_train, y_train, deg)
            r2 = r2_score(y_test, model(x_test))

            if not (np.any(model(x_smooth) < 0) or np.any(model(x_smooth) > max(y_athlete) * 1.05)):
                if r2 > best_r2:
                    best_r2 = r2
                    bestLine = model
                    best_deg = deg

        # print("FOR {} THE BEST DEGREE: {} WITH R2: {}".format(athlete_id, best_deg, best_r2))
        if best_r2 > 0.5:
            fig.add_trace(go.Scatter(x=x_smooth, y=bestLine(x_smooth), name="SMOOTH PERSON LINE"))
            # GOOD FIT RETURN
            return athlete_id, fig.to_dict(), bestLine, best_r2

        elif best_r2 == 0:
            # NO FIT RETURN
            return athlete_id, fig.to_dict(), None, best_r2


        else:
            fig.add_trace(go.Scatter(x=x_smooth, y=bestLine(x_smooth), name="SMOOTH PERSON LINE (POOR FIT)"))
            # POOR FIT RETURN
            return athlete_id, fig.to_dict(), bestLine, best_r2


    else:
        # NOT LONG ENOUGH RETURN
        return athlete_id, fig.to_dict(), None, 10


def create_groupGraph(inputList):
    colors = [
        '#3C4953', '#4A5E77', '#5A7492', '#6B8AAE', '#7FA5C7',
        '#556B83', '#4A6073', '#313D53', '#3C5266', '#4A657C',
        '#5A7C92', '#6B91A8', '#79A3B7', '#8EB6C8', '#5B7385',
        '#475A6F', '#314459', '#3A5772', '#4D6B88', '#678296',
        '#79A0AE', '#8EB3BB', '#5E7A8C', '#6A8A9C', '#88A2A7'
    ]

    print("MAKING GROUP GRAPH")
    fig = go.Figure()
    counter = 0
    for athlete_id in inputList:
        athlete_dataFrame = listOfDFs[athlete_id]

        x = athlete_dataFrame['age'].values

        poly_function = athleteLines[athlete_id]
        myLine = np.linspace(min(x), max(x), 100)

        athlete_name = athleteInfo.loc[athlete_id]['full_name']
        if poly_function is not None:
            fig.add_trace(
                go.Scatter(x=myLine, y=poly_function(myLine), name=athlete_name, customdata=[athlete_id] * len(myLine),
                           zorder=0, marker=dict(color=colors[counter % len(colors)])))
            counter += 1
    return fig


@callback(
    Output(component_id='athleteGraph', component_property='figure'),
    Output(component_id='my-output', component_property='children'),
    Input(component_id='athleteInput', component_property='value'),
    Input(component_id='everyoneGraph', component_property='clickData'), prevent_initial_call=True
)
def update_individual(athleteInput, clickData):
    trigger_id = ctx.triggered_id
    if trigger_id == 'everyoneGraph':
        athlete_id = clickData['points'][0]['customdata']
    else:
        athlete_id = NameToID([athleteInput])[0]

    if athlete_id in listOfAthletes:
        fig = athleteFigs[athlete_id]
        athlete_name = athleteInfo.loc[athlete_id]['full_name']
    else:
        fig = go.Figure()
        athlete_name = "ATHLETE NOT FOUND"
    return fig, athlete_name


@callback(
    Output(component_id='everyoneGraph', component_property='figure'),
    Input(component_id='everyoneGraph', component_property='clickData'),
    Input('everyoneGraph', 'figure'),
    Input('DropdownBox', 'value'),
    Input('athleteInput', 'value'),
    Input('age-slider', 'value'),
    Input('clubsDropdown', 'value')
)
def update_graph(clickData, fig, athleteDropdown, indivAthleteDropdown, rangeSlider, clubsDropdown):
    trigger_id = ctx.triggered_id

    if trigger_id == 'everyoneGraph' or trigger_id == 'athleteInput':
        greys = [
            '#8A8D8F', '#A6AAB2', '#B8C4CC', '#CED4DB', '#D9E0E5',
            '#9DAAB6', '#8C9FAF', '#6B788C', '#75889E', '#8498AF',
            '#A7B8CC', '#BFC9D6', '#CBD4DF', '#E6EBF2', '#9CAAB8',
            '#8A9AAB', '#6F7E8D', '#7D8A99', '#A4B2C0', '#C1CBD8',
            '#D6DEE5', '#E8EDF3', '#B0BBC8', '#C5CDD4', '#E1E5EB']

        if trigger_id == 'everyoneGraph':
            athlete_id = clickData['points'][0]['customdata']
        else:
            athlete_id = NameToID([indivAthleteDropdown])[0]

        patch_fig = Patch()

        for counter, trace in enumerate(fig['data']):
            aid = trace['customdata'][0]
            if aid == athlete_id:
                patch_fig['data'][counter]['marker']['color'] = 'green'
                patch_fig['data'][counter]['zorder'] = 5
            else:
                patch_fig['data'][counter]['marker']['color'] = greys[counter % len(greys)]
                patch_fig['data'][counter]['zorder'] = 1

        return patch_fig
    else:
        athletes = NameToID(athleteDropdown) if athleteDropdown else listOfAthletes
        filtered_athletes = clubFilter(athletes, clubsDropdown) if clubsDropdown else athletes
        return ageFilteredGraph(rangeSlider[0], rangeSlider[1], filtered_athletes)


@callback(Output('DropdownBox', 'options'),
          Input('age-slider', 'value'))
def update_dropdown(sliderValues):
    return idToName([athlete_id for athlete_id in listOfAthletes if
                     sliderValues[0] <= athleteInfo.loc[athlete_id]['birthyear'] < sliderValues[1]])


def ageFilteredGraph(minAge, maxAge, athletesList):
    filtered_athletes = athleteInfo[
        (athleteInfo.index.isin(athletesList)) &
        (athleteInfo['birthyear'] >= minAge) &
        (athleteInfo['birthyear'] < maxAge)
        ]

    filtered_athlete_ids = filtered_athletes.index.tolist()

    return create_groupGraph(filtered_athlete_ids)


def clubFilter(athleteList, club):
    filtered_athletes = athleteInfo[
        (athleteInfo.index.isin(athleteList)) &
        (athleteInfo['clubs'].apply(lambda x: club in x))
        ]
    return filtered_athletes.index.tolist()


if __name__ == '__main__':
    listOfAthletes, listOfDFs, listOfClubs, athleteInfo = get_data.returnData()

    groupLine = create_GroupLine()
    # calcIndividual2()
    concurrentIndividual()
    print("SIZE OF LIST OF ATHLETES: ", len(listOfAthletes))
    app = Dash()
    app.layout = html.Div([
        html.Div([
            html.Div("View Athlete:", style={'display': 'flex', 'align-items': 'center'}),
            dcc.Dropdown(idToName(listOfAthletes), id='athleteInput', placeholder='Select Athlete',
                         style={'flex-grow': '1'}),
            html.Div(style={'flex-grow': '1'})
        ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '10px', 'justify-content': 'flex-start',
                  'align-items': 'center'}
        ),
        dcc.RangeSlider(1950, 2010, step=1,
                        marks={1950: '1950', 1960: '1960', 1970: '1970', 1980: '1980', 1990: '1990', 2000: '2000',
                               2010: '2010'}, value=[2000, 2008], id='age-slider', pushable=1,
                        tooltip={"placement": "top", "always_visible": True}),
        html.Div([
            html.Content([dcc.Dropdown(idToName(listOfAthletes), id="DropdownBox", multi=True,
                                       placeholder="Select Athlete(s) To View")], style={'flex-grow': '1'})
            , html.Content([dcc.Dropdown(listOfClubs, id='clubsDropdown', placeholder="Select Club To View")],
                           style={'flex-grow': '1'})
        ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '10px', 'justify-content': 'space-evenly'}
        ),
        html.Br(),
        html.Div(id='my-output'),
        html.Div([
            dcc.Graph(id="everyoneGraph", figure=ageFilteredGraph(2000, 2008, listOfAthletes)),
            dcc.Graph(id='athleteGraph')
        ], style={'display': 'flex', 'justify-content': 'space-between'})])

    app.run(debug=True, use_reloader=False)
