# method overloading using *args
class CustomerService:
    def request(self, *args):
        if len(args) == 1:
            print(f"Handling general request: {args[0]}")
        elif len(args) == 2:
            print(f"Handling specific request: {args[0]} regarding {args[1]}")
        else:
            print("Invalid request.")

if __name__ == "__main__":
    service = CustomerService()
    service.request("Account closure")
    service.request("Card Issue", "Credit Card")