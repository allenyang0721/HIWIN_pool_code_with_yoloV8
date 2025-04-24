import math
class human:
    def __init__(self,name,age,gender):
        self.name=name
        self.gender=gender
        self.age=age

    def greet(self):
        print(F"你好,我是{self.name},我今年{self.age}歲,我是{self.gender}生")

class student(human):
    def __init__(self, name, age, gender,id,classroom,club):
        super().__init__(name, age, gender)
        self.id=id
        self.classroom=classroom
        self.club=club

    def greet(self):
         print(F"你好,我是{self.name},我今年{self.age}歲,我是{self.gender}生,我的學號是{self.id},班級是{self.classroom},我的社團叫做{self.club}")  

class teacher(human):
    def __init__(self, name, age, gender,subject):
        super().__init__(name, age, gender)
        self.subject=subject
    def greet(self):
         print(F"你好,我是{self.name},我今年{self.age}歲,我是一名{self.gender}性,我教的科目是{self.subject}")
       
class BankAccount:
    def __init__(self,balance):
        self.__balance=balance
        self.__balance=0
        self._simple_withdraw=5000

    def deposit(self,money):
        self.__balance=self.__balance+money

    def withdraw(self,money): 
        if   self.__balance-money>=0: 
            if money<self.simple_withdraw:
                self.__balance=self.__balance-money
                print(f"成功提款{money}元")
            else:
                print("提款金額超出單次上限")
        else:
            print("您的存款餘額不足")
    
    def get_balance(self):
        print(f"你的存款金額為{self.__balance}NTD")

#class vipaccount(BankAccount):



if __name__=="__main__":
    allen=BankAccount(3000)
    allen.deposit(100)
    BankAccount.get_balance(allen)




# if __name__=="__main__":
#     y_n=input("是否擁有銀行帳戶(Y/N)")
#     if y_n=='Y' or 'y':
#         inputaccount=input("請輸入帳號")
#         inputpassword=input("輸入提款密碼")
#     if y_n=='n'or'N':
#         newaccount=input("創建新帳號:")



# from abc import ABC,abstractmethod
# class Shape(ABC):
#     @abstractmethod
#     def area(self):
#         pass
#     @abstractmethod
#     def perimeter(self):
#         pass

# class triangle(Shape):
#     def __init__(self,long):
#         self.long=long


#     def counta(long):
#         b=long/2
#         a=b*(b*(math.sqrt(3)))/2
#         return a
#     def round(long):
#         c=long*3
#         return c
# if __name__=="__main__":
#     print("面積",triangle.counta(4))
#     print("周長",triangle.round(4))



        
    
# if __name__=="__main__":
    
#     allen=student("楊承恩",18,"男","D1352961","自控一乙","AiClub")
#     lin=student("林泓",19,"男","D1353000","自控一乙","AiClub")
#     boyki=teacher("謝南凱",45,"男","邏輯設計")
#     student.greet(allen)
#     student.greet(lin)
#     teacher.greet(boyki)



'''
class car:
    def __init__(self,brand,speed):
        self.brand =brand
        self.speed=speed
    def show(self):
        print("brand",self.brand,"speed",self.speed,"km/h")

class person:
    def __init__(self,name,age):
        self.age=age
        self.name=name
        
    def greet(self):
        print(f"hello,my name is {self.name} and i am {self.age} years old.")




if __name__=="__main__":
   
    mybank=BankAccount(3000)
    BankAccount.get_balance(mybank)
    BankAccount.deposit(mybank,300)
    BankAccount.get_balance(mybank)
    BankAccount.withdraw(mybank,500)
    BankAccount.get_balance(mybank)
'''   


'''
class Dog():
    def __init__(self,name,age,color):
        self.name=name
        self.age=age
        self.color=color
        pass
    def bark(self):
        print("他根本沒愛過你")

if __name__=="__main__":
    dog1= Dog("allen",3,"black")
'''




