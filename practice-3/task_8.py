cities = {
    'Kyiv': {
        'country': 'Ukraine',
        'population': '2.8 million',
        'fact': 'Capital of Ukraine'
    },
    'Tokyo': {
        'country': 'Japan',
        'population': '14 million',
        'fact': 'One of the most populous cities in the world'
    },
    'New York': {
        'country': 'USA',
        'population': '8.3 million',
        'fact': 'Known as the "Big Apple"'
    }
}

for city, info in cities.items():
    print(f"\nCity: {city}")
    for key, value in info.items():
        print(f"{key.capitalize()}: {value}")