teams = {
    'NEW YORK KNICKS': [22, 7, 6, 9, 45],
    'LOS ANGELES LAKERS': [20, 10, 5, 5, 40],
    'CHICAGO BULLS': [18, 8, 4, 6, 36]
}

for team, stats in teams.items():
    print(team, *stats)