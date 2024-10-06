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
