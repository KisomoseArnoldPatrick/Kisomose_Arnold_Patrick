class MobileApp:
    def notify(self):
        print("Mobile app notification sent.")
        
class EmailService:
    def notify(self):
        print("Email notification sent.")
        
class OnlineBanking(MobileApp, EmailService):
    pass
        
if __name__ == "__main__":
    # This will demonstrate the method resolution order (MRO)
    banking = OnlineBanking()
    banking.notify()