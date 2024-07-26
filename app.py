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
import warnings
from datetime import datetime
import math
from dash import Dash, dcc, html, Input, Output, callback
import dbDetails

listOfDFs = {}
listOfAthletes = []
athleteInfo = {}
athleteLines = {}
athleteFigs = {}
fit = {}
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
    newArray = []
    for i in inputArray:
        newArray.append(str(athleteInfo[i]['firstname'] + " " + athleteInfo[i]['lastname']))
    return newArray


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


def find_closest_performances(input_ages, x, y, num_closest):
    input_ages = np.asarray(input_ages)
    differences = np.abs(x[:, np.newaxis] - input_ages)

    closest_indices = np.argpartition(differences, num_closest, axis=0)[:num_closest, :]
    # weights = calc_weight(x, y)

    closest_performances = y[closest_indices]
    closest_difference = differences[closest_indices, np.arange(input_ages.size)]
    closest_weights = calc_weight(x[closest_indices], y[closest_indices])

    # Vectorized weight calculations
    exp_neg_diff = np.exp(-closest_difference)
    weighted_perf = exp_neg_diff * closest_weights * closest_performances
    weights_sum = exp_neg_diff * closest_weights

    # Compute the weighted mean for each input age
    mean = np.sum(weighted_perf, axis=0) / (np.sum(weights_sum, axis=0))  # Avoid division by zero

    return mean


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
    x = []
    y = []
    for athletes in listOfAthletes:
        athleteDF = listOfDFs[athletes]

        for age in athleteDF['age'].values:
            x.append(age)

        for wa_points in athleteDF['wa_points'].values:
            y.append(wa_points)

        # can end loop here if no need to visualise line creation

    poly_function = (calculateGroupBestFit(x, y))
    myLine = np.linspace(min(x), max(x), 100)

    return poly_function


def calc_weight(x_in, y_in):
    if len(x_in) > 0:
        w = np.ones(x_in.shape[0])

        w = np.zeros_like(x_in)
        w = groupLine(x_in) - y_in

        min_val = np.min(w)
        max_val = np.max(w)

        w = (1 - (w - min_val) / (max_val - min_val)) ** 3

        # w = np.exp(2 * (1 - (w - min_val) / (max_val - min_val)) - 2)
        # w[0] = min(1, 2 * w[0])
        # w[-1] = min(1, 2 * w[0])
        return w
    else:
        print("EMPTY ARRAY CALLED")


def calcIndividual2():
    time = datetime.now()

    goodLineCount = 0
    poorfitCount = 0
    nofitCount = 0
    notlongCount = 0
    for athlete_id in listOfAthletes:
        best_degree = 0
        best_r2 = 0
        athleteDF = listOfDFs[athlete_id]
        fig = go.Figure()
        if len(athleteDF) > 10:
            athleteDF['readablePerformance'] = athleteDF['performance'].apply(seconds_to_minutes_and_seconds)

            fig = px.scatter(
                athleteDF, x='age', y='wa_points', hover_data=['event', 'readablePerformance']
            )

            x_athlete = athleteDF['age'].to_numpy()
            y_athlete = athleteDF['wa_points'].to_numpy()
            yearsRunning = int(max(x_athlete) - min(x_athlete))

            x_smooth = np.linspace(min(x_athlete), max(x_athlete), yearsRunning * 24)
            y_smooth = list()
            polyvalue = groupLine(x_smooth)
            y_smooth = (polyvalue + 19 * find_closest_performances(x_smooth, x_athlete, y_athlete, 3)) / 20
            fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, marker=dict(color='red'), name="PERSON LINE"))

            fig.add_trace(go.Scatter(x=x_smooth, y=groupLine(x_smooth), name="GROUP LINE"))

            for deg in range(1, 10):
                x_train, x_test, y_train, y_test = train_test_split(x_smooth, y_smooth, test_size=0.3, random_state=100)
                model = np.polynomial.Polynomial.fit(x_train, y_train, deg)

                y_pred = model(x_test)
                r2 = r2_score(y_test, y_pred)
                if r2 > best_r2:
                    best_r2 = r2
                    bestLine = model
                    best_deg = deg

            print("FOR {} THE BEST DEGREE: {} WITH R2: {}".format(athlete_id, best_deg, best_r2))
            if best_r2 > 0.5:
                goodLineCount += 1
                fig.add_trace(go.Scatter(x=x_smooth, y=bestLine(x_smooth), name="SMOOTH PERSON LINE"))
                athleteFigs[athlete_id] = fig.to_dict()
                athleteLines[athlete_id] = bestLine

            elif best_r2 == 0:
                nofitCount += 1
                athleteFigs[athlete_id] = fig.to_dict()
                athleteLines[athlete_id] = None

            else:
                poorfitCount += 1
                fig.add_trace(go.Scatter(x=x_smooth, y=bestLine(x_smooth), name="SMOOTH PERSON LINE (POOR FIT)"))
                athleteFigs[athlete_id] = fig.to_dict()
                athleteLines[athlete_id] = bestLine

        else:
            notlongCount += 1
            athleteFigs[athlete_id] = fig.to_dict()
            athleteLines[athlete_id] = None


    # print("FOR {} NOT ENOUGH DATA POINTS".format(athlete_id))
    print("GOOD FITS: {}".format(goodLineCount))
    print("POOR FITS: {}".format(poorfitCount))
    print("NO FITS: {}".format(nofitCount))
    print("NOT BIG ENOUGH: {}".format(notlongCount))
    print("TOOK: {}".format(datetime.now() - time))


def calcIndividual():
    goodLineCount = 0
    poorfitCount = 0
    nofitCount = 0
    degreeCount = [0] * 30
    for athlete_id in listOfAthletes:
        best_degree = 0
        best_r2 = 0
        athletedf = listOfDFs[athlete_id]
        if len(athletedf) > 8:

            train_idx, test_idx = train_test_split(athletedf.index, test_size=0.25, random_state=104, shuffle=True)
            athletedf['split'] = 'train'
            athletedf.loc[test_idx, 'split'] = 'test'

            x = athletedf['age'].values
            y = athletedf['wa_points'].values
            x_train = x[train_idx]
            y_train = y[train_idx]
            x_test = x[test_idx]
            y_test = y[test_idx]

            w = calc_weight(x_train, y_train)

            fig = go.Figure()

            athletedf['readablePerformance'] = athletedf['performance'].apply(seconds_to_minutes_and_seconds)

            fig = px.scatter(
                athletedf, x='age', y='wa_points', color='split', hover_data=['event', 'readablePerformance']
            )

            yearsRunning = int(max(x) - min(x))
            d = max(yearsRunning, 8)
            d = min(yearsRunning, 15)

            for degree in range(1, d):
                # Perform polynomial regression
                # creating model
                firstModel = np.polynomial.Polynomial.fit(x_train, y_train, degree, w=w, domain=[min(x), max(x)])
                unweighedModel = np.polynomial.Polynomial.fit(x_train, y_train, degree)
                y_pred = unweighedModel(x)

                # checking for overfitting
                y_testing = firstModel(x_test)
                r2first = r2_score(y_test, y_testing, sample_weight=calc_weight(x_test, y_test))
                # print("R-squared score of test data for degree {} with outliers: {:.4f}".format(degree, r2first))

                # checking for outliers
                residuals = y - y_pred
                std_dev = np.std(residuals)
                threshold = 2 * std_dev
                mask = np.abs(residuals) <= threshold  # Efficient outlier detection using boolean mask

                x_filtered = x[mask]
                y_filtered = y[mask]
                # print("REMOVED {} outliers".format(len(X) - len(x_filtered)))

                # Modelling without outliers
                secondModel = np.polynomial.Polynomial.fit(x_filtered, y_filtered, degree,
                                                           w=calc_weight(x_filtered, y_filtered),
                                                           domain=[min(x), max(x)])
                y_testing2 = secondModel(x_filtered)
                r2second = r2_score(y_filtered, y_testing2, sample_weight=calc_weight(x_filtered, y_filtered))

                # print("R-squared score of test data for degree {} without outliers: {:.4f}".format(degree, r2second))

                myline = np.linspace(min(x), max(x), 100)
                # print("LINE DRAWN")
                if not (np.any(secondModel(myline) < 0) or np.any(secondModel(myline) > max(y))):
                    fig.add_trace(
                        go.Scatter(x=myline, y=secondModel(myline),
                                   name='Degree {} // NO OUTLIERS // WEIGHTED'.format(degree),
                                   line=dict(color="#ff0000"),
                                   legendrank=1 - r2second))

                if not (np.any(unweighedModel(myline) < 0) or np.any(unweighedModel(myline) > max(y))):
                    fig.add_trace(
                        go.Scatter(x=myline, y=unweighedModel(myline),
                                   name='Degree {} // UNWEIGHTED'.format(degree),
                                   line=dict(color=str("#cfff00"))))

                if not (np.any(firstModel(myline) < 0) or np.any(firstModel(myline) > max(y))):
                    fig.add_trace(
                        go.Scatter(x=myline, y=firstModel(myline),
                                   name='Degree  {} // WEIGHTED'.format(degree),
                                   line=dict(color="#00ff69"),
                                   legendrank=1 - r2first))

                if not (np.any(secondModel(myline) < 0) or np.any(secondModel(myline) > max(y))):
                    if r2second > best_r2:
                        bestLine = secondModel
                        best_r2 = r2second
                        best_degree = degree
            #         else:
            #             print("Worse Model")
            #     else:
            #         print("Degree {} not accepted", format(degree))
            #         print("Too low : {}".format(np.any(firstModel(myline) < 0)))
            #         print("Too high : {}".format(np.any(firstModel(myline) > 1400)))
            #         print("Too High for runner: {}".format(np.any(firstModel(myline) > max(y))))
            # #
            # print()
            # print("Best degree of polynomial:", best_degree)
            # print("Best R-squared score on test data:", best_r2)
            # print()
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
    calcIndividual2()
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
