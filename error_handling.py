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

        except ZeroDivisionError :
            print("Error: Division by zero is not allowed. Please enter a non-zero denominator.")
            continue
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            continue

        while True:
            choice = input("Do you want to try again? (yes/no): ").strip().lower()
            if choice in ['no', 'n']:
                print('Exiting the program.')
                return
            
            elif choice in ['yes', 'y']:
                print()
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")
                
if __name__ == "__main__":
    divide_numbers()