def get_valid_number(prompt):
    while True:
        try:
            number = float(input(prompt))
            return number
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            
def divide_numbers():
    while True:
        try:
            num1 = get_valid_number("Enter the first number (numerator): ")
            num2 = get_valid_number("Enter the second number (denominator): ")
            result = num1 / num2
            print(f"The result of {num1} divided by {num2} is {result}")
            break
        except ZeroDivisionError:
            print("Error: Division by zero is not allowed. Please enter a non-zero denominator.")   
            
if __name__ == "__main__":
    divide_numbers()

