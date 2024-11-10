import pprint
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import plotly.express as px
from multiprocessing import Pool
import get_data
from datetime import datetime
from dash import Dash, dcc, html, Input, Output, callback, ctx, Patch
import pandas as pd
import matplotlib.dates as mdates

athleteLines = {}
athleteFigs = {}
clubLines = {}
groupLine = None

listOfAthletes, listOfDFs, listOfClubs, athleteInfo, clubsDF = [], [], [], {}, {}


def seconds_to_minutes_and_seconds(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    if minutes > 60:
        hours = minutes // 60
        minutes -= hours * 60
        return f"{hours}:{minutes}:{remaining_seconds:.0f}"
    else:
        return f"{minutes}:{remaining_seconds:.2f}"


def calcFit(x, y):
    x = np.array(x)
    y = np.array(y)
    best_r2 = 0
    bestLine = None
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=101)
    # fig= go.Figure()
    fig = px.scatter(x=x, y=y)
    for deg in range(1, 7):
        model = np.polynomial.Polynomial.fit(x_train, y_train, deg)
        r2 = r2_score(y_test, model(x_test))
        x_graph = np.linspace(min(x), max(x), 100)
        # print("r2 = {}".format(r2))

        fig.add_trace(go.Scatter(x=x_graph, y=model(x_graph), name=deg))

        if not (np.any(model(x_graph) < 0) or np.any(model(x_graph) > max(y) * 1.05)):
            if r2 > best_r2:
                best_r2 = r2
                bestLine = model

    if best_r2 > 0:
        # print("BEST r2 = {}".format(best_r2))
        return bestLine, fig
    else:
        return None, fig


def create_GroupLine(x_data):
    def calculateGroupBestFit(x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=106, test_size=0.33, shuffle=True)

        best_degree = 0
        best_r2 = 0
        bestLine = None
        print("SIZE OF ARRAY {}".format(len(x)))

        for degree in range(1, 5):
            # Perform polynomial regression
            fitting = np.polyfit(x_train, y_train, degree)
            model = np.poly1d(fitting)
            y_testing = model(x_test)

            r2all = r2_score(y_test, y_testing)
            print("R-squared score of test data for degree {} with outliers: {:.4f}".format(degree, r2all))

            x_smooth = np.linspace(min(x), max(x), 200)

            if not (np.any(model(x_smooth) > max(y) * 1.05)):
                if r2all > best_r2:
                    best_r2 = r2all
                    best_degree = degree
                    bestLine = model

        print("BEST DEGREE IS: {}".format(best_degree))

        return bestLine

    start = datetime.now()
    x = []
    y = []

    for athlete in listOfAthletes:
        x.extend(listOfDFs[athlete][x_data].to_numpy())
        y.extend(listOfDFs[athlete]['wa_points'].to_numpy())


    poly_function = calculateGroupBestFit(x, y)

    print("FINISHED MAKING GROUPLINE {}".format(datetime.now() - start))

    return poly_function


def create_GroupLine2(athleteList, x_data):
    start = datetime.now()
    x = []
    y = []

    for athlete in athleteList:
        line, minn, maxx, raw = athleteLines[athlete][x_data]
        yearsrunning = int(maxx - minn)

        if yearsrunning > 0 and line is not None:
            x_tmp = np.linspace(minn, maxx, yearsrunning * 12)
            y_tmp = line(x_tmp)

            x.extend(x_tmp)
            y.extend(y_tmp)

    if len(x) > 3 and len(y) > 3:
        poly_function = calcFit(x, y)
    else:
        poly_function = None

    print("FINISHED MAKING GROUPLINE 2 {}".format(datetime.now() - start))

    return poly_function


from multiprocessing import Pool
from datetime import datetime


def process_individual_params(label, list_of_params, pool):
    """Helper function to process individual parameters (age, dec_date, etc.) and collect results."""
    nofitCount, poorfitCount, notlongCount, goodfitCount = 0, 0, 0, 0

    results = pool.starmap(calcIndivConcurrent, list_of_params)

    for result in results:
        athlete_id, fig, line, r2, minn, maxx, raw = result

        # Initialize athleteFigs[athlete_id] and athleteLines[athlete_id] if they don't exist
        if athlete_id not in athleteFigs:
            athleteFigs[athlete_id] = {}
        if athlete_id not in athleteLines:
            athleteLines[athlete_id] = {}

        # Update the data
        athleteFigs[athlete_id].update({label: fig})
        athleteLines[athlete_id].update({label: (line, minn, maxx, raw)})

        # Count fit quality
        if r2 > 0.5:
            goodfitCount += 1
        elif r2 == 0:
            nofitCount += 1
        elif r2 == 10:
            notlongCount += 1
        else:
            poorfitCount += 1

    print(f"{label.upper()}")
    print(f"GOOD FITS: {goodfitCount}")
    print(f"POOR FITS: {poorfitCount}")
    print(f"NO FITS: {nofitCount}")
    print(f"NOT BIG ENOUGH: {notlongCount}")


def concurrentIndividual():
    start = datetime.now()

    listofparam_age = [[athlete_id, listOfDFs[athlete_id], groupLine, 'age'] for athlete_id in listOfAthletes]
    listofparam_date = [[athlete_id, listOfDFs[athlete_id], groupLine, 'dec_date'] for athlete_id in listOfAthletes]

    # Use one Pool for both tasks
    with Pool(processes=8) as pool:
        process_individual_params('age', listofparam_age, pool)
        process_individual_params('dec_date', listofparam_date, pool)

    print(f"FINISHED CONCURRENT INDIVIDUALS IN {datetime.now() - start}")


def calcIndivConcurrent(athlete_id, athleteDF, group_line, x_data):
    def find_closest_performances_conc_2(x_linspace, x, y):
        x_linspace = np.asarray(x_linspace)

        def gaussian(x_gau, amp=1, mean=0, sigma=1):
            return amp * np.exp(-(x_gau - mean) ** 2 / (2 * sigma ** 2))

        def squewed_gaussian(x_gau, amp=1, mean=0, weight=1, sigma=1):
            return amp * np.exp(-0.5 * ((x_gau - mean) / (weight + (sigma * (x_gau- mean)))) ** 2)

        def calc_weight_conc2(x_in, y_in):
            if len(x_in) > 0:
                w = group_line(x_in) - y_in
                if w.size > 0:
                    w_min, w_max = np.min(w), np.max(w)
                    if w_min != w_max:
                        normalisation = (w - w_min) / (w_max - w_min)
                        w = gaussian(normalisation, amp = 1, mean = 0, sigma = 0.4)
                    else:
                        w = np.ones_like(w)
                else:
                    w = np.array([])

                return w
            else:
                print("EMPTY ARRAY CALLED")

        performance_weights = calc_weight_conc2(x, y)

        if performance_weights.size == 0:
            performance_weights = np.ones_like(x)  # Default to ones if empty

        averages = np.zeros(len(x_linspace))

        for i, bin_center in enumerate(x_linspace):
            # closeness_weights = gaussian(x, mean=bin_center, sigma=0.5)  # 1D array

            closeness_weights = squewed_gaussian(x, mean=bin_center, weight=-0.3, sigma=0.2)  # 1D array
            # print(closeness_weights)

            combined_weights = closeness_weights * performance_weights  # 1D array

            differences = np.abs(bin_center - x)
            closest = np.min(differences)

            aging = gaussian(closest, mean=0, sigma=2)

            averages[i] = aging * np.average(y, weights=combined_weights)

        return averages

    best_r2 = 0
    # athleteDF = listOfDFs[athlete_id]
    athleteDF['readablePerformance'] = athleteDF['performance_time'].apply(seconds_to_minutes_and_seconds)
    fig = go.Figure()

    if x_data == 'age':
        x_value = athleteDF['age'].tolist()
        fig.update_xaxes(title='Age')
    else:
        x_value = athleteDF['dec_date'].tolist()
        fig.update_xaxes(title='Year')

    age = athleteDF['age'].round(2).tolist()
    wa_points = athleteDF['wa_points'].tolist()
    events = athleteDF['event'].tolist()
    readable_performances = athleteDF['readablePerformance'].tolist()

    event_colors = {
        'Mile': '#dad7cd',  # Light Sage
        '5K': '#3a5a40',  # Olive Green
        '10K': '#a3b18a',  # Sage
        '400': '#344e41',  # Dark Green
        '800': '#588157',  # Moss Green
        '1500': '#a3b18a',  # Sage (distinct from 800)
        '3000': '#344e41',  # Dark Green (distinct from 1500)
        '5000': '#588157',  # Moss Green (distinct from 3000)
        '10000': '#dad7cd',  # Light Sage
        'HM': '#3a5a40',  # Olive Green
        'Mar': '#a3b18a',  # Sage
        '3000SC': '#344e41'  # Dark Green
    }

    colors = [event_colors[event] for event in athleteDF['event']]

    # Add a scatter plot trace with vectorized data
    fig.add_trace(go.Scattergl(
        x=x_value,
        y=wa_points,
        mode='markers',
        marker=dict(color=colors),
        text=[f"Event: {event}<br>Performance: {performance}<br>Age: {age}<br>Points: {points}" for
              event, performance, age, points in
              zip(events, readable_performances, age, wa_points)],
        hoverinfo='text',
        name='Athlete Performances'
    ))

    fig.update_yaxes(title='World Athletics Points')

    x_athlete = athleteDF[x_data].to_numpy()
    y_athlete = athleteDF['wa_points'].to_numpy()
    yearsRunning = int((max(x_athlete) - min(x_athlete)))

    if len(x_athlete) > 6 and yearsRunning > 0:

        if x_data == 'age':
            x_smooth = np.linspace(min(x_athlete), max(x_athlete), yearsRunning * 24)
        else:
            currentyear = (mdates.date2num(datetime.now()) / 365.25) + 1970
            x_smooth = np.linspace(min(x_athlete), currentyear, int((currentyear - min(x_athlete)) * 24))

        y_smooth = find_closest_performances_conc_2(x_smooth, x_athlete, y_athlete)
        fig.add_trace(
            go.Scattergl(x=x_smooth, y=y_smooth, marker=dict(color='blue'), name="ROLLING AVERAGE 2")
        )

        x_train, x_test, y_train, y_test = train_test_split(x_smooth, y_smooth, random_state=101)
        bestLine = None
        for deg in range(2, max(9, min(yearsRunning, 19))):
            model = np.polynomial.Polynomial.fit(x_train, y_train, deg)
            r2 = r2_score(y_test, model(x_test))

            if not (np.any(model(x_smooth) < 0) or np.any(model(x_smooth) > max(y_athlete) * 1.05)):
                if r2 > best_r2:
                    best_r2 = r2
                    bestLine = model

        performance_coordinates = pd.DataFrame({'x': x_smooth, 'y': y_smooth})

        # print("FOR {} THE BEST DEGREE: {} WITH R2: {}".format(athlete_id, best_deg, best_r2))
        if best_r2 > 0.5:
            fig.add_trace(
                go.Scattergl(x=x_smooth, y=bestLine(x_smooth), name="SMOOTH PERSON LINE", marker=dict(color='#bc4749')))
            # GOOD FIT RETURN
            return athlete_id, fig.to_dict(), bestLine, best_r2, min(x_athlete), max(x_athlete), performance_coordinates

        elif best_r2 == 0:
            # NO FIT RETURN
            return athlete_id, fig.to_dict(), None, best_r2, min(x_athlete), max(x_athlete), performance_coordinates

        else:
            fig.add_trace(go.Scatter(x=x_smooth, y=bestLine(x_smooth), name="SMOOTH PERSON LINE (POOR FIT)"))
            # POOR FIT RETURN
            return athlete_id, fig.to_dict(), bestLine, best_r2, min(x_athlete), max(x_athlete), performance_coordinates

    else:
        # NOT LONG ENOUGH RETURN
        return athlete_id, fig.to_dict(), None, 10, min(x_athlete), max(x_athlete), None


def concurrentClubs():
    start = datetime.now()

    listofparam_age = [[club, clubFilter(listOfAthletes, club), athleteLines, 'age', clubsDF.loc[clubsDF['club'] == club]] for club in listOfClubs]
    listofparam_date = [[club, clubFilter(listOfAthletes, club), athleteLines, 'dec_date', clubsDF[clubsDF['club'] == club]] for club in
                        listOfClubs]

    # Use one Pool for both tasks
    with Pool(processes=8) as pool:
        process_club_params('age', listofparam_age, pool)
        process_club_params('dec_date', listofparam_date, pool)

    print(f"FINISHED CONCURRENT INDIVIDUALS IN {datetime.now() - start}")


def process_club_params(label, list_of_params, pool):
    """Helper function to process individual parameters (age, dec_date, etc.) and collect results."""
    nofitCount, poorfitCount, notlongCount, goodfitCount = 0, 0, 0, 0

    results = pool.starmap(calcClubConcurrent, list_of_params)

    for result in results:
        club, clubLine = result
        if club is not None:
            if club not in clubLines:
                clubLines[club] = {}

            clubLines[club].update({label: clubLine})


def calcClubConcurrent(club, athleteList, athleteLines, label, clubsDF_):
    def create_clubLine(athleteList, x_data, clubname):
        x = []
        y = []
        for athlete in athleteList:
            line, start_running, end_running, raw = athleteLines[athlete][x_data]
            relations = clubsDF_[(clubsDF_['club'] == clubname) & (clubsDF_['athlete_id'] == athlete)]
            for _, relation in relations.iterrows():
                # pprint.pprint(relation)
                if x_data == 'dec_date':
                    minn = relation[x_data]['start_year']
                    maxx = relation[x_data]['end_year'] + 1

                    minn = max(minn, start_running)
                    maxx = min(maxx, end_running)
                else:
                    minn = start_running
                    maxx = end_running

                yearsatclub = int(maxx - minn)

                # print("ATHLETE ID: {} CLUB: {} START: {} END: {} ".format(athlete,clubname,minn,maxx))

                if yearsatclub > 0 and line is not None:
                    x_tmp = np.linspace(minn, maxx, yearsatclub * 12)
                    y_tmp = line(x_tmp)

                    x.extend(x_tmp)
                    y.extend(y_tmp)

        if len(x) > 3 and len(y) > 3:
            poly_function, figr = calcFit(x, y)
        else:
            poly_function, figr = None, None

        if poly_function is None:
            print("NO LINE FOR " + clubname + " FOR " + x_data)
            return None, None, None

        return poly_function, min(x), max(x)

    if athleteList and len(athleteList) > 1:
        validLine = any(athleteLines[athlete][label][0] for athlete in athleteList)
        if validLine:
            line, smallest, largest = create_clubLine(athleteList, label, club)
            clubLine = (line, smallest, largest)
        else:
            clubLine = (None, None, None)
    else:
        clubLine = None
    return club, clubLine


def clubFilter(athleteList, searchClub):
    dfClub = clubsDF.loc[clubsDF['club'] == searchClub]
    athletes = dfClub['athlete_id'].to_numpy()
    athleteList = np.array(athleteList)
    filtered_athletes = np.intersect1d(athletes, athleteList)

    return filtered_athletes.tolist()


def calcedData():
    global listOfAthletes, listOfDFs, listOfClubs, athleteInfo, clubsDF
    listOfAthletes, listOfDFs, listOfClubs, athleteInfo, clubsDF = get_data.returnData()

    global groupLine
    groupLine = create_GroupLine('age')
    concurrentIndividual()
    concurrentClubs()
    groupLine = create_GroupLine2(listOfAthletes, 'age')

    return (listOfAthletes, listOfDFs, listOfClubs, athleteInfo, clubsDF), (
    athleteLines, athleteFigs, clubLines, groupLine),
