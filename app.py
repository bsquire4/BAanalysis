import pprint
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import plotly.express as px
from multiprocessing import Pool
import calc_data
from datetime import datetime
from dash import Dash, dcc, html, Input, Output, callback, ctx, Patch

athleteLines = {}
athleteFigs = {}
clubLines = {}
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
    start = datetime.now()
    x = []
    y = []

    for athlete in listOfAthletes:
        x.extend(listOfDFs[athlete][x_data].to_numpy())
        y.extend(listOfDFs[athlete]['wa_points'].to_numpy())

    poly_function = calculateGroupBestFit(x, y)
    print("FINISHED MAKING GROUPLINE {}".format(datetime.now() - start))

    # fig = go.Figure()
    # x_line = np.linspace(min(x), max(x), 200)
    # fig.add_trace(go.Scattergl(x=x, y=y, mode = 'markers'))
    # fig.add_trace(go.Scattergl(x=x_line, y=poly_function(x_line), legendrank=100))
    #
    # fig.show()

    return poly_function


def create_GroupLine2(athleteList, x_data):
    start = datetime.now()
    x = []
    y = []

    for athlete in athleteList:
        line, minn, maxx = athleteLines[athlete][x_data]
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
    # fig = go.Figure()
    # fig.add_trace(go.Scattergl(x=x, y=y, mode='markers'))
    # x_line = np.linspace(min(x),max(x), 200)
    # fig.add_trace(go.Scattergl(x=x_line, y=poly_function(x_line), legendrank=100))
    # fig.show()

    print("FINISHED MAKING GROUPLINE 2 {}".format(datetime.now() - start))

    return poly_function


from multiprocessing import Pool
from datetime import datetime


def process_individual_params(label, list_of_params, pool):
    """Helper function to process individual parameters (age, dec_date, etc.) and collect results."""
    nofitCount, poorfitCount, notlongCount, goodfitCount = 0, 0, 0, 0

    results = pool.starmap(calcIndivConcurrent, list_of_params)

    for result in results:
        athlete_id, fig, line, r2, minn, maxx = result

        # Initialize athleteFigs[athlete_id] and athleteLines[athlete_id] if they don't exist
        if athlete_id not in athleteFigs:
            athleteFigs[athlete_id] = {}
        if athlete_id not in athleteLines:
            athleteLines[athlete_id] = {}

        # Update the data
        athleteFigs[athlete_id].update({label: fig})
        athleteLines[athlete_id].update({label: (line, minn, maxx)})

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
    with Pool(processes=16) as pool:
        process_individual_params('age', listofparam_age, pool)
        process_individual_params('dec_date', listofparam_date, pool)

    print(f"FINISHED CONCURRENT INDIVIDUALS IN {datetime.now() - start}")


def calcIndivConcurrent(athlete_id, athleteDF, group_line, x_data):

    def find_closest_performances_conc_2(x_linspace, x, y):
        x_linspace = np.asarray(x_linspace)

        def gaussian(x_gau, amp=1, mean=0, sigma=1):
            return amp * np.exp(-(x_gau - mean) ** 2 / (2 * sigma ** 2))

        def calc_weight_conc2(x_in, y_in):
            if len(x_in) > 0:
                w = group_line(x_in) - y_in
                if w.size > 0:
                    w_min, w_max = np.min(w), np.max(w)
                    if w_min != w_max:
                        w = np.exp(-5 * ((w - w_min) / (w_max - w_min)))
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
            closeness_weights = gaussian(x, mean=bin_center, sigma=0.5)  # 1D array
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

        x_smooth = np.linspace(min(x_athlete), max(x_athlete), yearsRunning * 24)
        y_smooth_2 = find_closest_performances_conc_2(x_smooth, x_athlete, y_athlete)
        fig.add_trace(
            go.Scattergl(x=x_smooth, y=y_smooth_2, marker=dict(color='blue'), name="ROLLING AVERAGE 2")
        )

        x_train, x_test, y_train, y_test = train_test_split(x_smooth, y_smooth_2, random_state=101)
        bestLine = None
        for deg in range(1, max(9, min(yearsRunning, 19))):
            model = np.polynomial.Polynomial.fit(x_train, y_train, deg)
            r2 = r2_score(y_test, model(x_test))

            if not (np.any(model(x_smooth) < 0) or np.any(model(x_smooth) > max(y_athlete) * 1.05)):
                if r2 > best_r2:
                    best_r2 = r2
                    bestLine = model

        # print("FOR {} THE BEST DEGREE: {} WITH R2: {}".format(athlete_id, best_deg, best_r2))
        if best_r2 > 0.5:
            fig.add_trace(
                go.Scattergl(x=x_smooth, y=bestLine(x_smooth), name="SMOOTH PERSON LINE", marker=dict(color='#bc4749')))
            # GOOD FIT RETURN
            return athlete_id, fig.to_dict(), bestLine, best_r2, min(x_athlete), max(x_athlete)

        elif best_r2 == 0:
            # NO FIT RETURN
            return athlete_id, fig.to_dict(), None, best_r2, min(x_athlete), max(x_athlete)

        else:
            fig.add_trace(go.Scatter(x=x_smooth, y=bestLine(x_smooth), name="SMOOTH PERSON LINE (POOR FIT)"))
            # POOR FIT RETURN
            return athlete_id, fig.to_dict(), bestLine, best_r2, min(x_athlete), max(x_athlete)

    else:
        # NOT LONG ENOUGH RETURN
        return athlete_id, fig.to_dict(), None, 10, min(x_athlete), max(x_athlete)


def create_groupGraph(inputList, x_data):
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
        poly_function, minn, maxx = athleteLines[athlete_id][x_data]
        myLine = np.linspace(minn, maxx, 100)

        athlete_name = athleteInfo.loc[athlete_id]['full_name']

        # Can change to Scattergl when zorder is added tp Scattergl
        # https://github.com/plotly/plotly.py/issues/4746
        # change the onClick function as well when changed
        if poly_function is not None:
            fig.add_trace(
                go.Scatter(x=myLine, y=poly_function(myLine), name=athlete_name,
                           customdata=[athlete_id] * len(myLine),
                           zorder=1, marker=dict(color=colors[counter % len(colors)])))
            counter += 1
    return fig


def show_groupLines(x_data):
    line1 = create_GroupLine(x_data)
    line2 = create_GroupLine2(listOfAthletes, x_data)

    if x_data == 'dec_data':
        x = np.linspace(2000, 2025, 400)
    else:
        x = np.linspace(10, 70, 400)
    y1 = line1(x)
    y2 = line2(x)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(x=x, y=y1, name="Original",
                     marker=dict(color='blue')))

    fig.add_trace(
        go.Scattergl(x=x, y=y2, name="New",
                     marker=dict(color='pink')))

    fig.show()


def concurrentClubs():
    start = datetime.now()

    listofparam_age = [[club, clubFilter(listOfAthletes,club),athleteLines, 'age', clubsDF] for club in listOfClubs]
    listofparam_date = [[club, clubFilter(listOfAthletes,club),athleteLines, 'dec_date', clubsDF] for club in listOfClubs]

    # Use one Pool for both tasks
    with Pool(processes=16) as pool:
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

            clubLines[club].update({label : clubLine})

def calcClubConcurrent(club, athleteList, athleteLines, label, clubsDF_):
    def create_clubLine(athleteList, x_data, clubname):
        x = []
        y = []
        for athlete in athleteList:
            line, start_running, end_running = athleteLines[athlete][x_data]
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



def create_clubsGraph(clubsList, x_data):
    fig = go.Figure()
    for club in clubsList:
        if clubLines[club][x_data]:
            # pprint.pprint(clubLines[club])
            polyLine, minn, maxx = clubLines[club][x_data]

            if polyLine:
                x = np.linspace(minn, maxx, 200)
                y = polyLine(x)
                fig.add_trace(go.Scatter(x=x, y=y, name=club, customdata=[club] * len(x)))

    return fig


@callback(
    Output(component_id='athleteGraph', component_property='figure'),
    Output(component_id='my-output', component_property='children'),
    Input(component_id='athleteInput', component_property='value'),
    Input(component_id='everyoneGraph', component_property='clickData'),
    Input('x_data_Input', 'value'),
    Input('my-output', 'children'),
    prevent_initial_call=True)
def update_individual(athleteInput, clickData, x_data, originalText):
    trigger_id = ctx.triggered_id

    if trigger_id == 'everyoneGraph':
        athlete_id = clickData['points'][0]['customdata']
    elif trigger_id == 'athleteInput':
        athlete_id = NameToID([athleteInput])[0]
    else:
        if originalText is None:
            return go.Figure(), ''
        else:
            athlete_id = NameToID([originalText])[0]

    if athlete_id in listOfAthletes:
        fig = athleteFigs[athlete_id][x_data]
        athlete_name = athleteInfo.loc[athlete_id]['full_name']
    else:
        fig = go.Figure()
        athlete_name = "ATHLETE NOT FOUND"
    return fig, athlete_name


@callback(
    Output('everyoneGraph', 'figure'),
    Input('everyoneGraph', 'clickData'),
    Input('everyoneGraph', 'figure'),
    Input('DropdownBox', 'value'),
    Input('athleteInput', 'value'),
    Input('age-slider', 'value'),
    Input('clubsDropdown', 'value'),
    Input('x_data_Input', 'value'),
)
def update_graph(clickData, fig, athleteDropdown, indivAthleteDropdown, rangeSlider, clubsDropdown, x_data):
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
        return ageFilteredGraph(rangeSlider[0], rangeSlider[1], filtered_athletes, x_data)


@callback(Output('DropdownBox', 'options'),
          Input('age-slider', 'value'))
def update_dropdown(sliderValues):
    return idToName([athlete_id for athlete_id in listOfAthletes if
                     sliderValues[0] <= athleteInfo.loc[athlete_id]['birthyear'] < sliderValues[1]])


@callback(Output('allClubGraph', 'figure'),
          Input('allClubsDropdown', 'value'),
          Input('x_data_Input', 'value'),
          prevent_initial_call=True)
def updateclubGraph(clubsDropdown, x_data):
    clubs = clubsDropdown if clubsDropdown else listOfClubs
    return create_clubsGraph(clubs, x_data)


@callback(Output('indivClubGraph', 'figure'),
          Input('allClubGraph', 'clickData'),
          Input('x_data_Input', 'value'),
          prevent_initial_call=True)
def updateClubGraphs(clickData, x_data):
    trigger_id = ctx.triggered_id
    if trigger_id == 'x_data_Input' and clickData is None:
        return go.Figure()

    clubName = clickData['points'][0]['customdata']

    return create_groupGraph(clubFilter(listOfAthletes, clubName), x_data)


def ageFilteredGraph(minAge, maxAge, athletesList, x_data):
    filtered_athletes = athleteInfo[
        (athleteInfo.index.isin(athletesList)) &
        (athleteInfo['birthyear'] >= minAge) &
        (athleteInfo['birthyear'] < maxAge)
        ]

    filtered_athlete_ids = filtered_athletes.index.tolist()

    return create_groupGraph(filtered_athlete_ids, x_data)


def clubFilter(athleteList, searchClub):
    dfClub = clubsDF.loc[clubsDF['club'] == searchClub]
    athletes = dfClub['athlete_id'].to_numpy()
    athleteList = np.array(athleteList)
    filtered_athletes = np.intersect1d(athletes, athleteList)

    return filtered_athletes.tolist()


if __name__ == '__main__':
    # x_data = 'age'
    athlete_data, calced_data = calc_data.calcedData()

    listOfAthletes, listOfDFs, listOfClubs, athleteInfo, clubsDF = athlete_data
    athleteLines,athleteFigs, clubLines, groupLine = calced_data

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div([
        dcc.Tabs([
            dcc.Tab(label='Athletes', children=[
                html.Div([
                    html.Div("View Athlete:", style={'display': 'flex', 'align-items': 'center'}),
                    dcc.Dropdown(idToName(listOfAthletes), id='athleteInput', placeholder='Select Athlete',
                                 style={'flex-grow': '1'}),
                    html.Div(style={'flex-grow': '1'}),
                    dcc.RadioItems(['dec_date', 'age'], 'age', inline=True, style={'flex-grow': '1'}, id='x_data_Input')
                ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '10px', 'justify-content': 'flex-start',
                          'align-items': 'center'}
                ),
                dcc.RangeSlider(1950, 2010, step=1,
                                marks={1950: '1950', 1960: '1960', 1970: '1970', 1980: '1980', 1990: '1990',
                                       2000: '2000',
                                       2010: '2010'}, value=[2000, 2008], id='age-slider', pushable=1,
                                tooltip={"placement": "top", "always_visible": True}),
                html.Div([
                    html.Content([dcc.Dropdown(idToName(listOfAthletes), id="DropdownBox", multi=True,
                                               placeholder="Select Athlete(s) To View")], style={'flex-grow': '1'}),
                    html.Content([dcc.Dropdown(listOfClubs, id='clubsDropdown', placeholder="Select Club To View")],
                                 style={'flex-grow': '1'})
                ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '10px', 'justify-content': 'space-evenly'}
                ),
                html.Br(),
                html.Div(id='my-output'),
                html.Div([
                    dcc.Graph(id="everyoneGraph", figure=ageFilteredGraph(2000, 2008, listOfAthletes, 'age')),
                    dcc.Graph(id='athleteGraph')
                ], style={'display': 'flex', 'justify-content': 'space-between'})]),
            dcc.Tab(label='Clubs', children=[
                html.Content([dcc.Dropdown(listOfClubs, id="allClubsDropdown", multi=True,
                                           placeholder="Select Club(s) To View")], style={'flex-grow': '1'}),
                html.Div([
                    dcc.Graph(id='allClubGraph', figure=create_clubsGraph(listOfClubs, 'age')),
                    dcc.Graph(id='indivClubGraph')
                ], style={'display': 'flex', 'justify-content': 'space-between'})

            ])
        ])
    ])

    app.run(debug=True, use_reloader=False)
