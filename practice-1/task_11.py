import math

def address_task():
    name = input("Введіть своє ім'я та прізвище: ")
    country = input("Введіть країну: ")
    postal_code = input("Введіть індекс: ")
    city = input("Введіть назву населеного пункту: ")
    street = input("Введіть назву вулиці: ")
    house_number = input("Введіть номер будинку: ")

    print(f"\nВаша адреса:\n{name}\n{country}\n{postal_code}\n{city}\n{street}\n{house_number}")

def temperature_conversion_task():
    celsius = float(input("\nВведіть температуру у градусах Цельсія: "))

    fahrenheit = 32 + (9 / 5) * celsius
    kelvin = celsius + 273.15

    print(f"\nТемпература у Фаренгейтах: {fahrenheit:^15.2f} °F")
    print(f"Температура у Кельвінах: {kelvin:^15.2f} K")

def event_duration_task():
    days = int(input("\nВведіть тривалість канікул у днях: "))

    hours = days * 24
    minutes = hours * 60
    seconds = minutes * 60

    print(f"\nТривалість:\n{hours:<10} годин\n{minutes:<10} хвилин\n{seconds:<10} секунд")

def distance_calculation_task():
    x1 = float(input("\nВведіть широту першої точки (градуси): "))
    y1 = float(input("Введіть довготу першої точки (градуси): "))
    x2 = float(input("Введіть широту другої точки (градуси): "))
    y2 = float(input("Введіть довготу другої точки (градуси): "))

    x1 = math.radians(x1)
    y1 = math.radians(y1)
    x2 = math.radians(x2)
    y2 = math.radians(y2)

    distance = 6371.032 * math.acos(math.sin(x1) * math.sin(x2) + 
                                      math.cos(x1) * math.cos(x2) * 
                                      math.cos(y1 - y2))

    print(f"\nВідстань між точками: {distance:>10.3f} км")

def main():
    print("Виберіть задачу, яку ви хочете виконати:")
    print("1. Введення адреси")
    print("2. Перетворення температури")
    print("3. Обчислення тривалості події")
    print("4. Обчислення відстані між точками")
    
    choice = input("Ваш вибір (1/2/3/4): ")

    if choice == '1':
        address_task()
    elif choice == '2':
        temperature_conversion_task()
    elif choice == '3':
        event_duration_task()
    elif choice == '4':
        distance_calculation_task()
    else:
        print("Неправильний вибір. Будь ласка, спробуйте ще раз.")

if __name__ == "__main__":
    main()
