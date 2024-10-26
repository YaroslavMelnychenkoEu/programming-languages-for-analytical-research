e2g = {
    'stork': 'storch',
    'hawk': 'falke',
    'woodpecker': 'specht',
    'owl': 'eule'
}

print("German for 'owl':", e2g['owl'])

e2g['sparrow'] = 'spatz'
e2g['eagle'] = 'adler'

print("\nDictionary:", e2g)
print("Keys:", list(e2g.keys()))
print("Values:", list(e2g.values()))