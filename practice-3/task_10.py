things = {
    'sword': 1,
    'shield': 1,
    'healing potion': 5,
    'gold coin': 50,
    'magic ring': 2,
    'arrows': 30,
    'torch': 3,
    'map': 1,
    'food ration': 10
}

print("Player's inventory:")
for item, quantity in things.items():
    print(f"{item}: {quantity}")