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

import warnings
from datetime import datetime
import math
from dash import Dash, dcc, html, Input, Output, callback, ctx, Patch
import dbDetails



conn = psycopg2.connect(
    user=dbDetails.DB_USER,
    password=dbDetails.DB_PASSWORD,
    host=dbDetails.DB_HOST,
    port=dbDetails.DB_PORT,
    database=dbDetails.DB_NAME)


def get_athletes():
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT athlete_id FROM athlete")
        athletes = cursor.fetchall()
        cursor.close()
    except Exception as e:
        print("ERROR GETTING ATHLETES")
        print(e)
    return [athlete[0] for athlete in athletes]


def get_performances():
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT athlete_id, date, event_title, performance_time, wa_points FROM athlete_performances")
        df = DataFrame(cursor.fetchall())
        df.columns = ['athlete_id', 'date', 'event', 'performance_time', 'wa_points']

        groupedDF = df.groupby(df.athlete_id)
        return groupedDF
    except Exception as e:
        print("ERROR GETTING ALL PERFORMANCES")
        print(e)


def get_info():
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT athlete_id, first_name, last_name, birthyear, club1_name, club2_name, club3_name FROM athlete_info")
        df = DataFrame(cursor.fetchall())
        df.columns = ['athlete_id', 'first_name', 'last_name', 'birthyear', 'club1', 'club2', 'club3']
        return df
    except Exception as e:
        print("ERROR GETTING INFO")
        print(e)


def returnData():
    print("RETURNDATA CALLED")
    listOfDFs = {}

    start = datetime.now()
    listOfAthletes = get_athletes()
    listOfAthletes = listOfAthletes[:100]
    athleteInfo = get_info()
    athleteInfo.set_index('athlete_id', inplace=True)
    groupsOfPerformance = get_performances()
    removalList = []

    print("GOT DATA IN TIME ", datetime.now() - start)
    counter = 0
    for athlete in listOfAthletes:
        counter += 1
        print(counter, "/", len(listOfAthletes))
        if athlete in athleteInfo.index:
            df = groupsOfPerformance.get_group(athlete).copy()
            dates = mdates.date2num(df['date'].values)
            birthdate = datetime(athleteInfo.loc[athlete]['birthyear'], 1, 1)
            ages = (dates - mdates.date2num(birthdate)) / 365.25
            df.loc[:, 'age'] = ages
            listOfDFs[athlete] = df
        else:
            removalList.append(athlete)

    listOfClubs = (
        athleteInfo['club1'].fillna('').tolist() +
        athleteInfo['club2'].fillna('').tolist() +
        athleteInfo['club3'].fillna('').tolist()
    )

    athleteInfo['full_name'] = athleteInfo['first_name'] + " " + athleteInfo['last_name']

    for ath in removalList:
        listOfAthletes.remove(ath)

    listOfClubs = list(set(filter(None, listOfClubs)))
    listOfClubs.sort()
    print("FINISHED GETTING DATA IN {}".format(datetime.now() - start))
    return listOfAthletes, listOfDFs,listOfClubs,athleteInfo
