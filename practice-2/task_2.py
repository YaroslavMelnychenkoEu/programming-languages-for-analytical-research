numbers = input("Введіть числа через пробіл: ").split()

numbers = list(map(int, numbers))
print("Сума чисел:", sum(numbers))