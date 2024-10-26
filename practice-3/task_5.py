pets = [
    {'name': 'Max', 'type': 'dog', 'owner': 'Alex'},
    {'name': 'Mittens', 'type': 'cat', 'owner': 'Maria'},
    {'name': 'Bubbles', 'type': 'fish', 'owner': 'Tom'}
]

for pet in pets:
    print(f"{pet['owner']} is the owner of a pet - a {pet['type']}.")