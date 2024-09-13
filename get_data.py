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

def get_athlete_clubs():
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT athlete_id, club_name, start_year, end_year FROM athlete_clubs_view")
        df = DataFrame(cursor.fetchall())
        df.columns = ['athlete_id', 'club', 'start_year', 'end_year']

        def create_dec_date(row):
            return {
                'dec_date': {
                    'start_year': row['start_year'],
                    'end_year': row['end_year']
                }
            }
        df = df.join(df.apply(create_dec_date, axis=1, result_type='expand'))

        def create_age(row):
            athletebirthyear = athleteInfo.loc[row['athlete_id']]['birthyear']
            return {
                'age': {
                    'start_year': int(row['start_year']) - int(athletebirthyear),
                    'end_year': int(row['end_year']) - int(athletebirthyear)
                }
            }

        df = df.join(df.apply(create_age, axis=1, result_type='expand'))

        print(df)
        return df
    except Exception as e:
        print("ERROR GETTING ATHLETE CLUBS")
        print(e)

def get_allclubs():
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM clubs")
        clubs = cursor.fetchall()
        clubs = np.array([t[0] for t in clubs])

        print(clubs)
        return clubs
    except Exception as e:
        print("ERROR GETTING ALL CLUB")
        print(e)

def get_info():
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT athlete_id, first_name, last_name, birthyear, region, gender FROM athlete_info")
        df = DataFrame(cursor.fetchall())
        df.columns = ['athlete_id', 'first_name', 'last_name', 'birthyear', 'region', 'gender']
        return df
    except Exception as e:
        print("ERROR GETTING INFO")
        print(e)


def returnData():
    print("RETURNDATA CALLED")
    listOfDFs = {}

    start = datetime.now()
    listOfAthletes = get_athletes()
    global athleteInfo
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
            df.loc[:, 'dec_date'] = (dates / 365.25) + 1970
            listOfDFs[athlete] = df
        else:
            removalList.append(athlete)

    listOfClubs = get_allclubs()

    clubsDF = get_athlete_clubs()

    athleteInfo['full_name'] = athleteInfo['first_name'] + " " + athleteInfo['last_name']

    for ath in removalList:
        listOfAthletes.remove(ath)

    listOfClubs.sort()
    print("FINISHED GETTING DATA IN {}".format(datetime.now() - start))
    return listOfAthletes, listOfDFs,listOfClubs,athleteInfo, clubsDF
