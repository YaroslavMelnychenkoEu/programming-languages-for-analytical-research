digits = input("Введіть 5 цифр через пробіл: ").split()

digit_list = list(digits)

reversed_list = digit_list[::-1]

result_number = ''.join(reversed_list)
print("Число із зворотного списку:", result_number)