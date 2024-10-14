import math

# Координати Чернівців (широта, довгота)
che_lat = 48.2923752098617
che_lon = 25.929201368386956

# Координати Ірпеня (широта, довгота)
irpin_lat = 50.50452889278587
irpin_lon = 30.24427032663506

x1 = math.radians(che_lat)
y1 = math.radians(che_lon)
x2 = math.radians(irpin_lat)
y2 = math.radians(irpin_lon)

distance = 6371.032 * math.acos(math.sin(x1) * math.sin(x2) + 
                                  math.cos(x1) * math.cos(x2) * 
                                  math.cos(y1 - y2))

print("{:>10.3f} км".format(distance))