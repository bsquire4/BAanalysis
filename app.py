import pprint

import numpy
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
import warnings
from datetime import datetime
import math
from dash import Dash, dcc, html, Input, Output, callback
import dbDetails
import statistics

listOfDFs = {}
listOfAthletes = []
athleteInfo = {}
athleteLines = {}
athleteFigs = {}
fit = {}
groupLine = None


def seconds_to_minutes_and_seconds(seconds):
    if seconds is None:
        return None
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        if minutes > 60:
            hours = minutes // 60
            minutes -= hours * 60
            return f"{hours}:{minutes}:{remaining_seconds:.0f}"
        else:
            return f"{minutes}:{remaining_seconds:.2f}"


def idToName(inputArray):
    newArray = []
    for i in inputArray:
        newArray.append(str(athleteInfo[i]['firstname'] + " " + athleteInfo[i]['lastname']))
    return newArray


def find_closest_performances(input_age, x, y, num_closest):
    # Calculate the absolute differences between the input age and all ages in the dataset
    differences = np.abs(x - input_age)

    # Find the indices of the smallest differences
    closest_indices = np.argsort(differences)[:num_closest]

    # Retrieve the corresponding ages and performances
    closest_ages = x[closest_indices]
    closest_performances = y[closest_indices]
    closest_difference = differences[closest_indices]

    mean = 0
    for i in range(num_closest):
        mean = (mean + (np.exp(-closest_difference[i]) * closest_performances[i])) / (
            np.exp(-closest_difference[i]) + 1)

    return mean


def NameToID(inputArray):
    if inputArray is None:
        return None
    else:
        newArray = []
        for input in inputArray:
            for athlete_id, info in athleteInfo.items():
                # Construct the full name from the dictionary
                full_name = info['firstname'] + " " + info['lastname']
                # Check if it matches the provided athlete name
                if full_name == input:
                    newArray.append(athlete_id)
                # If no match is found, return None or raise an exception
        return newArray


def get_data():
    conn = psycopg2.connect(
        user=dbDetails.DB_USER,
        password=dbDetails.DB_PASSWORD,
        host=dbDetails.DB_HOST,
        port=dbDetails.DB_PORT,
        dbname=dbDetails.DB_NAME
    )
    cursor = conn.cursor()

    query = sql.SQL("""
                    SELECT (event_title, date, performance_time, wa_points) FROM athlete_performances WHERE athlete_id = (%s)
                            """)

    get_athletes = sql.SQL("""
                    SELECT (athlete_id) FROM athlete
                                """)

    get_athlete_info = sql.SQL("""
                    SELECT (first_name, last_name, birthyear) FROM athlete_info WHERE athlete_id = (%s)
                                       """)

    cursor.execute(get_athletes)
    athleteList = cursor.fetchall()

    for athlete in athleteList:
        cursor.execute(get_athlete_info, (athlete,))
        output = cursor.fetchone()
        id = int(athlete[0])

        firstname, lastname, birthyear = output[0].replace('(', '').replace(')', '').split(',')
        athlete_info = ({
            'firstname': firstname,
            'lastname': lastname,
            'birthyear': int(birthyear)
        })
        athleteInfo[id] = athlete_info

        listOfAthletes.append(id)
        athlete_data = []
        cursor.execute(query, (athlete,))
        output = cursor.fetchall()

        for out in output:
            out = out[0].strip('()').split(',')

            if out[3] != '':
                performance_data = {
                    'event': str(out[0]),
                    'date': out[1],
                    'performance': float(out[2]),
                    'wa_points': float(out[3])
                }
                athlete_data.append(performance_data)

        athleteDataFrame = pd.DataFrame(athlete_data)

        dates = mdates.date2num(athleteDataFrame['date'].values)

        x = []
        for xi in dates:
            birthyear = datetime(athlete_info['birthyear'], month=1, day=1)
            x.append(int(xi - mdates.date2num(birthyear)) / 365.25)

        athleteDataFrame = athleteDataFrame.assign(age=x)

        listOfDFs[id] = athleteDataFrame
        # print(firstname + " " + lastname + " - " + str(id))


def calculateGroupBestFit(x, y):
    if len(x) > 6:
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=104, test_size=0.25, shuffle=True)

        best_degree = 0
        best_r2 = 0
        bestX = []
        bestY = []

        for degree in range(1, 7):
            # Perform polynomial regression
            mymodel = np.polyfit(x_train, y_train, degree)
            poly_function = np.poly1d(mymodel)
            y_pred = poly_function(x)

            r2all = r2_score(y, y_pred)
            print("R-squared score of test data for degree {} with outliers: {:.4f}".format(degree, r2all))
            residuals = y - y_pred

            # Compute the standard deviation of the residuals
            std_dev = np.std(residuals)

            # A common criterion: outliers are points where the residual is more than 2 standard deviations away
            threshold = 3 * std_dev
            outlier_indices = np.where(np.abs(residuals) > threshold)[0]

            x_filtered = np.delete(x, outlier_indices)
            y_filtered = np.delete(y, outlier_indices)

            print("REMOVED {} outliers".format(len(x) - len(x_filtered)))

            coeffs_refined = np.polyfit(x_filtered, y_filtered, degree)
            poly_func_refined = np.poly1d(coeffs_refined)

            y_pred2 = poly_func_refined(x_filtered)
            r2inliers = r2_score(y_filtered, y_pred2)
            print(
                "R-squared score of test data for degree {} without outliers: {:.4f}".format(degree, r2inliers))

            if r2inliers > best_r2:
                best_r2 = r2inliers
                best_degree = degree
                bestX = x_filtered
                bestY = y_filtered

        return np.polyfit(bestX, bestY, best_degree)


def create_GroupLine():
    x = []
    y = []
    for athletes in listOfAthletes:
        athleteDF = listOfDFs[athletes]

        for age in athleteDF['age'].values:
            x.append(age)

        for wa_points in athleteDF['wa_points'].values:
            y.append(wa_points)

        # can end loop here if no need to visualise line creation

    poly_function = np.poly1d(calculateGroupBestFit(x, y))
    myLine = np.linspace(min(x), max(x), 100)

    return poly_function


def calc_weight(x_in, y_in):
    poly = np.poly1d(groupLine)
    residuals = (y_in - poly(x_in))

    w = (residuals - np.min(residuals)) / (np.max(residuals) - np.min(residuals))
    weights = w ** 2

    weights[0] = min(1, weights[0] * 2)
    weights[-1] = min(1, weights[-1] * 2)

    return weights


def calcIndividual():
    goodLineCount = 0
    poorfitCount = 0
    nofitCount = 0
    degreeCount = [0] * 30
    for athlete_id in listOfAthletes:
        best_degree = 0
        best_r2 = 0
        athletedf = listOfDFs[athlete_id]

        fig = go.Figure()

        if len(athletedf) > 10:
            x_df = athletedf['age'].to_numpy()
            y_df = athletedf['wa_points'].to_numpy()
            yearsRunning = int(max(x_df) - min(x_df))
            d = min(max(yearsRunning, 8), 15)

            # add data to smoothen line
            x_additional = np.linspace(min(x_df), max(x_df), yearsRunning * 2)
            y_additional = list()
            for i in x_additional:
                y_additional.append(float((groupLine(i) + find_closest_performances(i, x_df, y_df, 5)) / 2))

            newdf = pd.DataFrame({
                'event': [None] * len(x_additional),  # Initialize with None or any other placeholder
                'date': [None] * len(x_additional),  # Initialize with None or any other placeholder
                'performance': [0] * len(x_additional),  # Initialize with None or any other placeholder
                'wa_points': np.array(y_additional),
                'age': np.array(x_additional),
                'split': ['smoothening data'] * len(x_additional)  # Repeat the string for each row
            })
            athletedf['split'] = None
            print(len(athletedf))
            print(len(newdf))
            athletedf = pd.concat([athletedf, newdf], ignore_index=True)

            x_df = athletedf['age'].to_numpy()
            y_df = athletedf['wa_points'].to_numpy()

            train_idx, test_idx = train_test_split(athletedf.index, test_size=0.25, random_state=104, shuffle=True)

            def splitDataTrain(value):
                return 'train' if pd.isna(value) else value

            athletedf['split'] = athletedf['split'].apply(splitDataTrain)

            def splitDataTest(value):
                return 'test' if pd.isna(value) else value

            athletedf.loc[test_idx, 'split'] = np.where(
                athletedf.loc[test_idx, 'split'].isna(), 'test', athletedf.loc[test_idx, 'split']
            )

            athletedf['readablePerformance'] = athletedf['performance'].apply(seconds_to_minutes_and_seconds)
            print(len(athletedf))

            fig = px.scatter(athletedf, x='age', y='wa_points', color='split',
                             hover_data=['event', 'readablePerformance'])
            x_train = x_df[train_idx]
            y_train = y_df[train_idx]
            x_test = x_df[test_idx]
            y_test = y_df[test_idx]

            for degree in range(1, d):
                # creating model
                firstModel = np.polynomial.Polynomial.fit(x_train, y_train, degree, w=calc_weight(x_train, y_train),
                                                          domain=[min(x_df), max(x_df)])
                unweighedModel = np.polynomial.Polynomial.fit(x_train, y_train, degree)

                # checking for overfitting
                y_testing = firstModel(x_test)
                r2first = r2_score(y_test, y_testing, sample_weight=calc_weight(x_test, y_test))
                # print("R-squared score of test data for degree {} with outliers: {:.4f}".format(degree, r2first))

                # checking for outliers
                y_pred = unweighedModel(x_df)
                residuals = y_df - y_pred
                std_dev = np.std(residuals)
                threshold = 2 * std_dev
                mask = np.abs(residuals) <= threshold  # Efficient outlier detection using boolean mask

                x_filtered = x_df[mask]
                y_filtered = y_df[mask]
                # print("REMOVED {} outliers".format(len(X) - len(x_filtered)))
                (x_filteredtrain, x_filteredtest,
                 y_filteredtrain, y_filteredtest) = train_test_split(x_filtered, y_filtered, test_size=0.25,
                                                                     random_state=42, shuffle=True)

                # Modelling without outliers
                secondModel = np.polynomial.Polynomial.fit(x_filteredtrain, y_filteredtrain, degree,
                                                           w=calc_weight(x_filteredtrain, y_filteredtrain),
                                                           domain=[min(x_filteredtrain), max(x_filteredtrain)])
                y_testing2 = secondModel(x_filteredtest)

                r2second = r2_score(y_filteredtest, y_testing2,
                                    sample_weight=calc_weight(x_filteredtest, y_filteredtest))

                # print("R-squared score of test data for degree {} without outliers: {:.4f}".format(degree, r2second))

                myline = np.linspace(min(x_df), max(x_df), 100)

                if not (np.any(firstModel(myline) < 0) or np.any(firstModel(myline) > max(y_df))):
                    fig.add_trace(go.Scatter(x=myline, y=firstModel(myline),
                                             name='Degree  {} // WEIGHTED'.format(degree),
                                             line=dict(color="#00ff69"), legendrank=1 - r2first))

                if not (np.any(unweighedModel(myline) < 0) or np.any(unweighedModel(myline) > max(y_df))):
                    fig.add_trace(go.Scatter(x=myline, y=unweighedModel(myline),
                                             name='Degree {} // UNWEIGHTED'.format(degree),
                                             line=dict(color=str("#cfff00"))))

                if not (np.any(secondModel(myline) < 0) or np.any(secondModel(myline) > max(y_df))):
                    fig.add_trace(go.Scatter(x=myline, y=secondModel(myline),
                                             name='Degree {} // NO OUTLIERS // WEIGHTED'.format(degree),
                                             line=dict(color="#ff0000"), legendrank=1 - r2second))

                    if r2second > best_r2:
                        bestLine = secondModel
                        best_r2 = r2second
                        best_degree = degree

        if best_r2 == 0:
            nofitCount += 1
            athleteFigs[athlete_id] = fig.to_dict()
            athleteLines[athlete_id] = None
        else:
            if best_r2 < 0.5:
                poorfitCount += 1
            else:
                goodLineCount += 1
            degreeCount[best_degree - 1] += 1
            athleteFigs[athlete_id] = fig.to_dict()
            athleteLines[athlete_id] = bestLine

    print("NO FIT: {} / {}".format(nofitCount, len(listOfAthletes)))
    print("POOR FIT: {} / {}".format(poorfitCount, len(listOfAthletes)))
    print("GOOD FIT: {} / {}".format(goodLineCount, len(listOfAthletes)))
    for i in range(len(degreeCount)):
        print("DEGREE {}: {}".format(i + 1, degreeCount[i]))


def create_groupGraph(inputList):
    colors = [
        '#4D5A65', '#5D758E', '#7292B4', '#85AFD9', '#9BCDEB',
        '#6D82A3', '#5C7291', '#404E66', '#4D6780', '#5A7A99',
        '#7197B4', '#87B0D1', '#95C5DD', '#B2D9ED', '#6C8FA9',
        '#59748D', '#405A73', '#4C6887', '#61809C', '#7DA2B7',
        '#95C1D1', '#B2D3DD', '#7E94A4', '#8AA1B2', '#A8C3CC'
    ]

    print("MAKING GROUP GRAPH")
    fig = go.Figure()
    counter = 0
    for athlete_id in inputList:
        athlete_dataFrame = listOfDFs[athlete_id]

        x = athlete_dataFrame['age'].values
        y = athlete_dataFrame['wa_points'].values

        # print("WRITING LINE :" + str(athlete_id))
        poly_function = athleteLines[athlete_id]
        myLine = np.linspace(min(x), max(x), 100)

        athlete_name = str(athleteInfo[athlete_id]['firstname'] + " " + athleteInfo[athlete_id]['lastname'])
        if poly_function is not None:
            # poly_function = np.poly1d(poly_function)
            fig.add_trace(
                go.Scatter(x=myLine, y=poly_function(myLine), name=athlete_name, customdata=[athlete_id] * len(myLine),
                           marker=dict(color=colors[counter % len(colors)]), zorder=0))
            counter += 1
            # print("LINE WRITTEN")
    return fig


@callback(
    Output(component_id='athleteGraph', component_property='figure'),
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value'),
    Input(component_id='everyoneGraph', component_property='clickData'), prevent_initial_call=True
)
def update_individual(textInput, clickData):
    athleteID = 0
    if clickData is not None:
        athleteID = clickData['points'][0]['customdata']
    elif textInput is not None:
        athleteID = int(textInput)
    if athleteID in listOfAthletes:
        print(athleteID)
        fig = athleteFigs[athleteID]
        athlete_name = str(athleteInfo[int(athleteID)]['firstname'] + " " + athleteInfo[int(athleteID)]['lastname'])
    else:
        fig = go.Figure()
        name = "ATHLETE NOT FOUND"
    return fig, athlete_name


@callback(
    Output(component_id='everyoneGraph', component_property='figure'),
    Input(component_id='everyoneGraph', component_property='clickData'),
    Input(component_id='everyoneGraph', component_property='figure'),
    Input('DropdownBox', 'value'), prevent_initial_call=True

)
def update_graph_colours(clickData, graph, dropDownValues):
    greys = [
        '#8A8D8F', '#A6AAB2', '#B8C4CC', '#CED4DB', '#D9E0E5',
        '#9DAAB6', '#8C9FAF', '#6B788C', '#75889E', '#8498AF',
        '#A7B8CC', '#BFC9D6', '#CBD4DF', '#E6EBF2', '#9CAAB8',
        '#8A9AAB', '#6F7E8D', '#7D8A99', '#A4B2C0', '#C1CBD8',
        '#D6DEE5', '#E8EDF3', '#B0BBC8', '#C5CDD4', '#E1E5EB']

    counter = 0
    if dropDownValues is not None:
        newfig = create_groupGraph(NameToID(dropDownValues))
    else:
        newfig = create_groupGraph(listOfAthletes)

    if clickData is not None:
        for trace in newfig['data']:
            if trace['customdata'][0] != clickData['points'][0]['customdata']:
                trace['marker']['color'] = greys[counter % len(greys)]
                trace['zorder'] = 0
            else:
                trace['marker']['color'] = 'purple'
                trace['zorder'] = 5
            counter += 1

    return newfig


if __name__ == '__main__':
    print("THen this")
    get_data()
    print("now that")
    groupLine = create_GroupLine()
    calcIndividual()
    print("this")

    app = Dash()
    app.layout = html.Div([
        html.Div([
            "Input: ",
            dcc.Input(id='my-input', placeholder='Input AthleteID', type='number')
        ]),
        dcc.Dropdown(idToName(listOfAthletes), id="DropdownBox", multi=True),
        html.Br(),
        html.Div(id='my-output'),
        html.Div([
            dcc.Graph(id="everyoneGraph", figure=create_groupGraph(listOfAthletes)),
            dcc.Graph(id='athleteGraph')
        ], style={'display': 'flex', 'justify-content': 'space-between'})])

    app.run(debug=True, use_reloader=False)
