import calc_data
import numpy as np
import pprint



def clubFilter(athleteList, searchClub):
    dfClub = clubsDF.loc[clubsDF['club'] == searchClub]
    athletes = dfClub['athlete_id'].to_numpy()
    athleteList = np.array(athleteList)
    filtered_athletes = np.intersect1d(athletes, athleteList)

    return filtered_athletes.tolist()

def clubImprovement(club, label):
    groupList = clubFilter(listOfAthletes,club)

    totalimprovment = 0
    athletecount = 0
    for athlete in groupList:
        line, minn, maxx = athleteLines[athlete][label]
        if line:
            values = clubsDF.loc[(clubsDF['club'] == club) & (clubsDF['athlete_id'] == athlete),label].values
            print(athleteInfo.loc[athlete]['full_name'])
            pprint.pprint(values)
            for value in values:
                starting = value['start_year']
                ending = value['end_year']

                startvalue = line(max(starting, minn))
                endvalue = line(min(ending, maxx))

                improvement = ((endvalue - startvalue) / startvalue) * 100
                print(improvement)
                totalimprovment += improvement
                athletecount += 1

    avgimprovment = totalimprovment / athletecount
    print("Average improvment of group {} is {}".format(label, avgimprovment))

def groupImprovement(athleteList, label, start, end):
    totalimprovment = 0
    athletecount = 0

    for athlete in athleteList:
        line, minn, maxx = athleteLines[athlete][label]
        if line:
            startvalue = line(max(start, minn))
            endvalue = line(min(end, maxx))

            improvement = ((endvalue - startvalue) / startvalue) * 100
            totalimprovment += improvement
            athletecount += 1

    avgimprovment = totalimprovment / athletecount
    print("Average improvment of group from age {} to {} is {}".format(start,end,avgimprovment))

def lineimprovemtn(line, startvalue, endvalue):
    print(line)
    if line:
        start = line(startvalue)
        end = line(endvalue)
        improvement = (end - start) / start * 100
        return improvement
    else:
        return 0




if __name__ == '__main__':
    athlete_data, calced_data = calc_data.calcedData()

    listOfAthletes, listOfDFs, listOfClubs, athleteInfo, clubsDF = athlete_data
    athleteLines, athleteFigs, clubLines, groupLine = calced_data

    clubImprovement('Birmingham Uni', 'age')
    clubImprovement('Loughborough Students', 'age')

    print("BIRMINGHAM UNI")
    groupImprovement(clubFilter(listOfAthletes, 'Birmingham Uni'), 'age', 18.5, 22)

    print("LOUGHBOROUGH UNI")
    groupImprovement(clubFilter(listOfAthletes, 'Loughborough Students'), 'age', 18.5, 22)

    print("BIRMINGHAM UNI (group Line)")
    lineimprovemtn(clubLines['Birmingham Uni']['age'], 18.5,22)

    print("LOUGHBOROUGH UNI (group Line)")
    lineimprovemtn(clubLines['Loughborough Students']['age'], 18.5,22)

