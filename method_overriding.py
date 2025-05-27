class BankAccount:
    def __init__(self, account_number, balance=0):
        self.account_number = account_number
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        print(f"Deposited {amount}. New balance is {self.balance}.")

    def withdraw(self, amount):
        if amount > self.balance:
            print("Insufficient funds.")
        else:
            self.balance -= amount
            print(f"Withdrew {amount}. New balance is {self.balance}.")
            
    def display_balance(self):
        print(f"Account {self.account_number} has balance: {self.balance}")
        
class SavingsAccount(BankAccount):
    def withdraw(self, amount):
        if amount > 2000000:
            print("Withdrawal limit exceeded for Savings Account.")
        else:
            super().withdraw(amount)
            
if __name__ == "__main__":
    # Create a BankAccount instance
    account = BankAccount("3202010104")
    account.deposit(5500)
    

    # Create a SavingsAccount instance
    savings_account = SavingsAccount("3202110105", 20000)
    savings_account.deposit(50000)
    
    # demonstrate method overriding
    account.withdraw(2000)
    savings_account.withdraw(3000000)

    