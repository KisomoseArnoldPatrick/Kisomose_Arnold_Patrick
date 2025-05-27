class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.id = None
        
    def display_info(self):
        print(f"Name: {self.name}")
        print(f"Age: {self.age}")
        if self.id:
            print(f"ID: {self.id}")
        
class Student(Person):
    def __init__(self, name, age, student_id, course):
        super().__init__(name, age)
        self.id = student_id
        self.course = course

    def display_info(self):
        super().display_info()
        print(f"Course: {self.course}")

class Lecturer(Person):
    def __init__(self, name, age, employee_id, department, salary):
        super().__init__(name, age)
        self.id = employee_id
        self.department = department
        self.salary = salary
        self.course_units = []

    def display_info(self):
        super().display_info()
        print(f"Department: {self.department}")
        print(f"Salary: {self.salary}")
        print(f"Course Units: {', '.join(self.course_units)}")

class Staff(Person):
    def __init__(self, name, age, employee_id, position, salary):
        super().__init__(name, age)
        self.id = employee_id
        self.position = position
        self.salary = salary

    def display_info(self):
        super().display_info()
        print(f"Position: {self.position}")
        print(f"Salary: {self.salary}")
        
if __name__ == "__main__":

    print('=== Student Information ===')
    student = Student("Alice", 20, "22/U/10017/PS", "Computer Science")
    student.display_info()

    print('\n=== Lecturer Information ===')
    lecturer = Lecturer("Dr. Smith", 45,"2012/MAK/12", "Mathematics", 80000)
    lecturer.course_units = ["Calculus", "Linear Algebra"]
    lecturer.display_info()

    print('\n=== Staff Information ===')
    staff = Staff("John Doe", 30, "2023/MAK/100", "Administrative Assistant", 40000)
    staff.display_info()
