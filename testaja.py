class Person(object):
       
    # Constructor
    def __init__(self, name):
        self.name = name
   
    # To get name
    def getName(self):
        return self.name
   
    # To check if this person is an employee
    def isEmployee(self):
        return False
   
   
# Inherited or Subclass (Note Person in bracket)
class Employee(Person):

    def __init__(self, name, salary, post):
        self.salary = salary
        self.post = post

        # invoking the __init__ of the parent class 
        Person.__init__(self, name)
   
    # Here we return true
    def isEmployee(self):
        return True

# Inherited or Subclass (Note Person in bracket)
class Manager(Person):

    def __init__(self, name, salary):
        self.salary = salary

        # invoking the __init__ of the parent class 
        Person.__init__(self, name)
   
    # Here we return true
    def isEmployee(self):
        return True
   
# Driver code
emp = Employee("Geek1", 400, "Intern")  # An Object of E 
emp1 = Manager("Geek2", 3000) # An Object of Employee

arrPerson = []
arrPerson.append(emp)
arrPerson.append(emp1)

print(arrPerson[0].getName())
print(arrPerson[1].getName())