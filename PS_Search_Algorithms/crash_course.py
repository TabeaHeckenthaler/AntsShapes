# parent class
class student:
    def __init__(self, age=19, subject='physics', gender='f'):
        # attributes
        self.age = age
        self.subject = subject
        self.gender = gender

    # methods
    def change_subject(self, new_subject):
        self.subject = new_subject

    def birthday(self):
        self.age = self.age + 1



#
#
# # child class (inherit parent class)
# class Weizmann_student(student):
#     def __init__(self):
#         super().__init__()
#         self.dorm = 'Clore'
#
#     def eat_at_SanMartin(self):
#         return
#
#     def change_subject(self, new_subject):
#         print('Hello')
#         self.subject = new_subject
#
#
# class HU_student(student):
#     def __init__(self):
#         super().__init__()
#         self.dorm = 'Clore'
#
#     def eat_at_SanMartin(self):
#         return
#
#     def change_subject(self, new_subject):
#         print('Hi')
#         self.subject = new_subject
#
#
# # Ctrl + Shift + F: Find all places with # TODO Rotation students
#
# # DEBUGGER:
# # F7: Jump into function
# # F8: Execute on line
# # F9: Continue until next debug point
