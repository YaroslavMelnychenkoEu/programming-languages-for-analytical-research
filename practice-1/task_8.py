value_in_celsius = 25

value_in_fahrenheit = 32 + 9/5 * value_in_celsius
value_in_kelvin = value_in_celsius + 273.15

print("{:^15} {:^15}".format("Шкала", "Температура"))
print("{:^15} {:^15.2f}".format("Цельсій", value_in_celsius))
print("{:^15} {:^15.2f}".format("Фаренгейт", value_in_fahrenheit))
print("{:^15} {:^15.2f}".format("Кельвін", value_in_kelvin))