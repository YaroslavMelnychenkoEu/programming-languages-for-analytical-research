languages = {
    'Python': 'Guido van Rossum',
    'JavaScript': 'Brendan Eich',
    'Java': 'James Gosling',
    'C++': 'Bjarne Stroustrup'
}

for language, developer in languages.items():
    print(f"My favorite programming language is {language}. It was created by {developer}.")

languages.pop('Java')
print("\nUpdated languages dictionary:")
print(languages)