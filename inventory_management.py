class InventoryManagement:
    def __init__(self):
        self.inventory = {}

    def add_item(self):
        item_name = input("Enter item name: ").lower()
        if item_name in self.inventory:
            print(item_name, "already exists in the inventory. Use update option instead.")
            return
        try:
            price = int(input("Enter item price: "))
            quantity = int(input("Enter item quantity: "))
            
            if quantity <= 0 or price <= 0:
                print("Quantity and price cannot be negative.")
                return
            
            self.inventory[item_name] = {
                "price": price,
                "quantity": quantity
            }
            print(f"Added {item_name} to inventory with price {price} and quantity {quantity}.")
        except ValueError:
            print("Invalid input. Please enter positive values for price and quantity.")

    def remove_item(self):
        item_name = input("Enter item name to remove: ").lower()
        if item_name in self.inventory:
            confirm = input(f"Are you sure you want to remove {item_name}? (yes/no): ").lower()
            if confirm != 'yes':
                 del self.inventory[item_name]
                 print(f"Removed {item_name} from inventory.")
            else:
                print("Item removal cancelled.")
               
        else:
            print(f"{item_name} not found in inventory.")
            return
        

    def search_item(self):
        search_item = input("Enter item name to search: ").lower()
        found_items = []
        for item in self.inventory:
            if search_item in item:
                found_items.append(item)
                
        if found_items:
            print(f"\nFound {len(found_items)} item(s) matching '{search_item}':")
            print(f"{'Item':<20} {'Price':<10} {'Quantity':<10}")
            print("-" * 50)
            for item in found_items:
                item_details = self.inventory[item]
                print(f"{item:<20} {item_details['price']:<10} {item_details['quantity']:<10}")
        else:
            print(f"No items found matching '{search_item}'.")
            
    def update_item(self):
        item_name = input("Enter item name to update: ").lower()
        if item_name in self.inventory:
            print(f"Current quantity of {item_name} is {self.inventory[item_name]['quantity']}")
            print(f"Current price of {item_name} is {self.inventory[item_name]['price']}")
            try:
                price = int(input("Enter new item price: "))
                quantity = int(input("Enter new item quantity: "))
                
                if quantity <= 0 or price <= 0:
                    print("Quantity and price cannot be negative.")
                    return
                
                self.inventory[item_name] = {
                    "price": price,
                    "quantity": quantity
                }
                print(f"Updated {item_name} in inventory with new price {price} and quantity {quantity}.")
            except ValueError:
                print("Invalid input. Please enter positive values for price and quantity.")
        else:
            print(f"{item_name} not found in inventory.")
            
    def display_inventory(self):
        # if self.inventory == {}: or
        if not self.inventory:
            print("Inventory is empty.")
            return
        
        print(f"\n{'Item':<20} {'Price':<10} {'Quantity':<10}")
        print("-" * 50)
        for item, details in self.inventory.items():
            print(f"{item:<20} {details['price']:<10} {details['quantity']:<10}")
    
    def run(self):
        menu_options = {
            '1': {'name': 'Add Item', 'function': self.add_item},
            '2': {'name': 'Remove Item', 'function': self.remove_item},
            '3': {'name': 'Search Item', 'function': self.search_item},
            '4': {'name': 'Update Item', 'function': self.update_item},
            '5': {'name': 'Display Inventory', 'function': self.display_inventory},
            '6': {'name': 'Exit', 'function': None}
        }
        while True:
            print("\nInventory Management System")
            print("-" * 30)
            for key, option in menu_options.items():
                print(f"{key}. {option['name']}")

            choice = input("Select an option (1-6): ")
            
            if choice in menu_options:
                if choice == '6':
                    print("Exiting the program.")
                    break
                    
                menu_options[choice]['function']()
                
            else:
                print("Invalid choice. Please select a valid option.")
                
if __name__ == "__main__":
    inventory_management = InventoryManagement()
    inventory_management.run()