value_in_meters = 1000

meters_to_inches = value_in_meters * 39.3701  # 1 метр = 39.3701 дюймів
meters_to_feet = value_in_meters * 3.28084   # 1 метр = 3.28084 фута
meters_to_yards = value_in_meters * 1.09361  # 1 метр = 1.09361 ярда
meters_to_miles = value_in_meters * 0.000621371  # 1 метр = 0.000621371 миль

print("Відстань в метрах: {} м".format(value_in_meters))
print("Відстань у дюймах: {:.2f} in".format(meters_to_inches))
print("Відстань у футах: {:.2f} ft".format(meters_to_feet))
print("Відстань у ярдах: {:.2f} yd".format(meters_to_yards))
print("Відстань у милях: {:.6f} mi".format(meters_to_miles))