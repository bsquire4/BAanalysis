import pprint

import pandas as pd
import psycopg2
from psycopg2 import sql
import matplotlib.pyplot as plt
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


def get_data():
    try:
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

            firstname, lastname, birthyear = output[0].replace('(', '').replace(')', '').split(',')
            athlete_info = ({
                'firstname': firstname,
                'lastname': lastname,
                'birthyear': int(birthyear)
            })
            athleteInfo[athlete[0]] = athlete_info

            listOfAthletes.append(int(athlete[0]))
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

            listOfDFs[athlete[0]] = athleteDataFrame
    except Exception as e:
        print("ERROR IMPORTING DATA: " + str(e))


def calculateGroupBestFit(x, y):
    if len(x) > 6:
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=104, test_size=0.25, shuffle=True)

        best_degree = 0
        best_r2 = 0
        bestX = []
        bestY = []

        for degree in range(1, 7):
            # Perform polynomial regression
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
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

                    r2 = 0.9 * r2inliers + 0.1 * r2all

                    firstderiv = poly_function.deriv()
                    roots = firstderiv.roots
                    secondderiv = firstderiv.deriv()
                    # Filter for minima
                    minima = []
                    for root in roots:
                        if secondderiv(root) > 0:  # Positive second derivative -> minimum
                            minima.append((root, poly_function(root)))

                    # Find the minimum point (if any)
                    min_point = (1, 1)
                    max_point = (1, 1)
                    if minima:
                        min_point = min(minima, key=lambda x: x[1])  # Minimize by y-value

                    maxima = []
                    for root in roots:
                        if secondderiv(root) < 0:  # Negative second derivative -> maximum
                            maxima.append((root, poly_function(root)))

                    # Find the maximum point (if any)
                    if maxima:
                        max_point = max(maxima, key=lambda x: x[1])  # Maximize by y-value

                    try:
                        if int(min_point[1]) > 0 and int(max_point[1]) < 1400 and int(max_point[1]) < max(y):
                            if r2 > best_r2:
                                best_r2 = r2
                                best_degree = degree
                                bestX = x_filtered
                                bestY = y_filtered
                    except:
                        print("IMAGINARY !!")
                except np.RankWarning:
                    print("not enough data")

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


groupLine = None

def calc_weight(x_in, y_in):
    poly = np.poly1d(groupLine)
    w = np.ones(x_in.shape[0])

    for i in range(len(w)):
        w[i] = poly(x_in[i]) - y_in[i]

    min_val = np.min(w)
    max_val = np.max(w)

    w = (1 - (w - min_val) / (max_val - min_val)) ** 2
    return w


def calculateBestFit(x, y):
    if len(x) > 8:
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=105, test_size=0.25, shuffle=True)

        best_r2 = 0
        bestLine = None

        w = calc_weight(x_train, y_train)

        for degree in range(1, 7):
            # Perform polynomial regression

            # creating model
            mymodel = np.polynomial.Polynomial.fit(x_train, y_train, degree, w=w)
            poly_function = np.poly1d(mymodel)
            mymodelUnweighted = np.polynomial.Polynomial.fit(x_train, y_train, degree)
            unweighted_poly_function = np.poly1d(mymodelUnweighted)
            y_pred = mymodel(x)

            # checking for overfitting
            y_testing = mymodel(x_test)
            r2all = r2_score(y_test, y_testing, sample_weight=calc_weight(x_test, y_test))
            # print("R-squared score of test data for degree {} with outliers: {:.4f}".format(degree, r2all))

            # checking for outliers
            residuals = y - y_pred
            std_dev = np.std(residuals)
            threshold = 2 * std_dev
            outlier_indices = np.where(np.abs(residuals) > threshold)[0]
            x_filtered = np.delete(x, outlier_indices)
            y_filtered = np.delete(y, outlier_indices)
            # print("REMOVED {} outliers".format(len(X) - len(x_filtered)))

            # Modelling without outliers
            w1 = calc_weight(x_filtered, y_filtered)
            mynewModel = np.polyfit(x_filtered, y_filtered, degree, w=w1)
            new_poly_function = np.poly1d(mynewModel)
            y_pred2 = new_poly_function(x_filtered)
            r2New = r2_score(y_filtered, y_pred2, sample_weight=calc_weight(x_filtered, y_filtered))
            # print("R-squared score of test data for degree {} without outliers: {:.4f}".format(degree, r2New))

            myline = np.linspace(min(x), max(x), 100)
            if not (np.any(new_poly_function(myline) < 0) or np.any(new_poly_function(myline) > 1400) or np.any(
                new_poly_function(myline) > max(y))):
                if r2New > best_r2:
                    bestLine = mynewModel
                    best_r2 = r2New

        if best_r2 == 0:
            return None
        else:
            return np.poly1d(bestLine)


def seconds_to_minutes_and_seconds(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    if minutes > 60:
        hours = minutes // 60
        minutes -= hours * 60
        return f"{hours}:{minutes}:{remaining_seconds:.0f}"
    else:
        return f"{minutes}:{remaining_seconds:.2f}"


def create_individualGraph():
    for athlete_id in listOfAthletes:
        best_degree = 0
        best_r2 = 0
        athletedf = listOfDFs[athlete_id]
        if len(athletedf) > 8:

            train_idx, test_idx = train_test_split(athletedf.index, test_size=0.25, random_state=104, shuffle=True)
            athletedf['split'] = 'train'
            athletedf.loc[test_idx, 'split'] = 'test'

            X = athletedf['age']
            y = athletedf['wa_points']
            X_train = athletedf.loc[train_idx]['age']
            y_train = athletedf.loc[train_idx]['wa_points']
            X_test = athletedf.loc[test_idx]['age']
            y_test = athletedf.loc[test_idx]['wa_points']

            w = calc_weight(X_train.to_numpy(), y_train.to_numpy())

            fig = go.Figure()

            athletedf['readablePerformance'] = athletedf['performance'].apply(seconds_to_minutes_and_seconds)

            fig = px.scatter(
                athletedf, x='age', y='wa_points', color='split', hover_data=['event', 'readablePerformance']
            )

            for degree in range(1, 7):
                # Perform polynomial regression
                # creating model
                mymodel = np.polynomial.Polynomial.fit(X_train.values, y_train.values, degree, w=w)
                poly_function = np.poly1d(mymodel)
                mymodelUnweighted = np.polynomial.Polynomial.fit(X_train.values, y_train.values, degree)
                unweighted_poly_function = np.poly1d(mymodelUnweighted)
                y_pred = mymodel(X.values)

                # checking for overfitting
                y_testing = mymodel(X_test.values)
                r2all = r2_score(y_test, y_testing)
                print("R-squared score of test data for degree {} with outliers: {:.4f}".format(degree, r2all))

                # checking for outliers
                residuals = y.values - y_pred
                std_dev = np.std(residuals)
                threshold = 3 * std_dev
                outlier_indices = np.where(np.abs(residuals) > threshold)[0]
                x_filtered = np.delete(X, outlier_indices)
                y_filtered = np.delete(y, outlier_indices)
                # print("REMOVED {} outliers".format(len(X) - len(x_filtered)))

                # Modelling without outliers
                mynewModel = np.polyfit(x_filtered, y_filtered, degree, w=calc_weight(x_filtered, y_filtered))
                new_poly_function = np.poly1d(mynewModel)
                y_pred2 = new_poly_function(x_filtered)
                r2New = r2_score(y_filtered, y_pred2)
                print("R-squared score of test data for degree {} without outliers: {:.4f}".format(degree, r2New))
                myline = np.linspace(min(X), max(X), 100)
                print("LINE DRAWN")
                if not (np.any(new_poly_function(myline) < 0) or np.any(new_poly_function(myline) > 1400)):
                    fig.add_trace(
                        go.Scatter(x=myline, y=new_poly_function(myline),
                                   name='Degree {} // NO OUTLIERS // WEIGHTED'.format(degree),
                                   line=dict(color="#ff0000"),
                                   legendrank=1 - r2New))

                if not (np.any(mymodelUnweighted(myline) < 0) or np.any(mymodelUnweighted(myline) > 1400)):
                    fig.add_trace(
                        go.Scatter(x=myline, y=mymodelUnweighted(myline),
                                   name='Degree {} // UNWEIGHTED'.format(degree),
                                   line=dict(color=str("#cfff00")),
                                   legendrank=1 - r2all))

                if not (np.any(mymodel(myline) < 0) or np.any(mymodel(myline) > 1400)):
                    fig.add_trace(
                        go.Scatter(x=myline, y=mymodel(myline),
                                   name='Degree  {} // WEIGHTED'.format(degree), line=dict(color="#00ff69")))

                if not (
                    np.any(new_poly_function(myline) < 0) or np.any(new_poly_function(myline) > 1400) or np.any(
                    new_poly_function(myline) > max(y))):
                    if r2New > best_r2:
                        bestLine = mynewModel
                        best_r2 = r2New
                        best_degree = degree

            print()
            print("Best degree of polynomial:", best_degree)
            print("Best R-squared score on test data:", best_r2)
            print()

        if best_r2 == 0:
            athleteFigs[athlete_id] = fig.to_dict()
            athleteLines[athlete_id] = None
        else:
            athleteFigs[athlete_id] = fig.to_dict()
            athleteLines[athlete_id] = bestLine


def show_individualGraph(athlete_id):
    if athlete_id in listOfAthletes:
        athlete_dataFrame = listOfDFs[athlete_id]

        x = athlete_dataFrame['age']
        y = athlete_dataFrame['wa_points']

        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=104, test_size=0.25, shuffle=True)

        best_degree = 0
        best_r2 = 0

        plt.figure(figsize=(8, 6))

        figs = go.Figure()
        figs.add_trace(go.Scatter(x=x_train, y=y_train, name='training data', mode="markers"))
        figs.add_trace(go.Scatter(x=x_test, y=y_test, name='test data', mode="markers"))

        for degree in range(1, 7):
            # Perform polynomial regression
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    mymodel = np.polyfit(x_train, y_train, degree)
                    poly_function = np.poly1d(mymodel)
                    y_pred = poly_function(x)

                    r2all = r2_score(y, y_pred)
                    print("R-squared score of test data for degree {} with outliers: {:.4f}".format(degree, r2all))

                    residuals = y - y_pred

                    # Compute the standard deviation of the residuals
                    std_dev = np.std(residuals)

                    # A common criterion: outliers are points where the residual is more than 3 standard deviations away
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

                    r2 = 0.9 * r2inliers + 0.1 * r2all
                    firstderiv = poly_function.deriv()
                    roots = firstderiv.roots
                    secondderiv = firstderiv.deriv()
                    # Filter for minima
                    minima = []
                    for root in roots:
                        if secondderiv(root) > 0:  # Positive second derivative -> minimum
                            minima.append((root, poly_function(root)))

                    # Find the minimum point (if any)
                    min_point = (1, 1)
                    max_point = (1, 1)
                    if minima:
                        min_point = min(minima, key=lambda x: x[1])  # Minimize by y-value

                    maxima = []
                    for root in roots:
                        if secondderiv(root) < 0:  # Negative second derivative -> maximum
                            maxima.append((root, poly_function(root)))

                    # Find the maximum point (if any)
                    if maxima:
                        max_point = max(maxima, key=lambda x: x[1])  # Maximize by y-value

                    try:
                        if int(min_point[1]) > 0 and int(max_point[1]) < 1400 and int(max_point[1]) < max(y):
                            myline = np.linspace(min(x), max(x), 100)
                            plt.plot(myline, poly_function(myline), label='Degree {}'.format(degree))

                            # mycontinuedline = np.linspace(max(x), mdates.date2num(datetime.today()) - 1970,100)
                            # plt.plot(mdates.num2date(mycontinuedline), poly_function(mycontinuedline), label=str(athlete_id) + " prediction", linestyle='dashed')

                            if r2 > best_r2:
                                best_r2 = r2
                                best_degree = degree
                    except:
                        print("IMAGINARY !!")
                except np.RankWarning:
                    print("not enough data")

            # Calculate R-squared score on test set

        figs.show()
        # Add labels and title
        plt.xlabel('Age')
        plt.gca().set_ybound(0, 1400)

        plt.ylabel('WA Points')
        plt.title('Polynomial Regression Analysis')
        plt.legend()

        # Show plot
        print()
        print("Best degree of polynomial:", best_degree)
        print("Best R-squared score on test data:", best_r2)
        print()
        plt.grid(True)
        plt.show()

    fig = plt.figure()
    linesPerYear = []


def create_groupGraph():
    print("MAKING GROUP GRAPH")
    fig = go.Figure()
    highestAge = 2024
    for athlete_id in listOfAthletes:
        athlete_dataFrame = listOfDFs[athlete_id]

        x = athlete_dataFrame['age'].values
        y = athlete_dataFrame['wa_points'].values

        print("WRITING LINE :" + str(athlete_id))
        poly_function = athleteLines[athlete_id]
        myLine = np.linspace(min(x), max(x), 100)

        athlete_name = str(athleteInfo[athlete_id]['firstname'] + " " + athleteInfo[athlete_id]['lastname'])
        if poly_function is not None:
            poly_function = np.poly1d(poly_function)
            fig.add_trace(
                go.Scatter(x=myLine, y=poly_function(myLine), name=athlete_name, customdata=[athlete_id] * len(myLine),
                           marker=dict(color='blue'), zorder=0))
            print("LINE WRITTEN")

    return fig


def show_GroupLine():
    fig = go.Figure()
    x = []
    y = []
    counter = 0
    for athletes in listOfAthletes:
        counter = counter + 1
        athleteDF = listOfDFs[athletes]

        for age in athleteDF['age'].values:
            x.append(age)

        for wa_points in athleteDF['wa_points'].values:
            y.append(wa_points)

        # can end loop here if no need to visualise line creation

        poly_function = np.poly1d(calculateGroupBestFit(x, y))
        myLine = np.linspace(min(x), max(x), 100)

        try:
            fig.add_trace(
                go.Scatter(x=myLine, y=poly_function(myLine), name="Iteration " + str(counter),
                           line=dict(color='rgb(67,67,' + str(counter * 10) + ')')))
        except Exception as e:
            print("ERROR DRAWING GROUP LINE OF BEST FIT")

    return fig


app = Dash(__name__)


@callback(
    Output(component_id='athleteGraph', component_property='figure'),
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value'),
    Input(component_id='everyoneGraph', component_property='hoverData')
)
def update_individual(textInput, hoverData):
    if hoverData is not None:
        pprint.pprint(hoverData)
        athleteID = hoverData['points'][0]['customdata']
        fig = athleteFigs[athleteID]
        return fig, str(athleteInfo[athleteID]['firstname'] + " " + athleteInfo[athleteID]['lastname'])
    else:
        if textInput in listOfAthletes:
            print(textInput)
            fig = athleteFigs[textInput]
            name = str(athleteInfo[int(textInput)]['firstname'] + " " + athleteInfo[int(textInput)]['lastname'])
        else:
            fig = go.Figure()
            name = "ATHLETE NOT FOUND"
        return fig, name


@callback(
    Output(component_id='everyoneGraph', component_property='figure'),
    Input(component_id='everyoneGraph', component_property='clickData'),
    Input(component_id='everyoneGraph', component_property='figure')
)
def update_graph_colours(clickData, graph):
    if clickData is not None:
        for trace in graph['data']:
            if trace['customdata'][0] != clickData['points'][0]['customdata']:
                trace['marker']['color'] = 'grey'
                trace['zorder'] = 0
            else:
                trace['marker']['color'] = 'purple'
                trace['zorder'] = 5
    return graph


if __name__ == '__main__':
    get_data()
    groupLine = create_GroupLine()
    create_individualGraph()
    pprint.pprint(athleteLines)
    app.layout = html.Div([
        html.Div([
            "Input: ",
            dcc.Input(id='my-input', placeholder='Input AthleteID', type='number')
        ]),
        html.Br(),
        html.Div(id='my-output'),
        html.Div([
            dcc.Graph(id="everyoneGraph", figure=create_groupGraph()),
            dcc.Graph(id='athleteGraph')
        ], style={'display': 'flex', 'justify-content': 'space-between'})])

    app.run(debug=True)
