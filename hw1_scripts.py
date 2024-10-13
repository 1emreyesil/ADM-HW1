

#**********PROBLEM 1**********

# Introduction_Say_"Hello, World!"_With_Python:

variable= "Hello, World!"
print(variable)

# Introduction_Python_If-Else:


import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(input().strip())

    if (n%2==0 and (n>2 and n<5)):
        print("Not Weird")
    elif (n%2==0 and (n>6 and n<20)):
        print("Weird")
    elif (n%2==0 and n>20):
        print("Not Weird")
    else :
        print("Weird")

# Introduction_Arithmetic_Operators:

a = int(input())
b = int(input())

print(a+b)
print(a-b)
print(a*b)

# Introduction_Python:Division:

if __name__ == '__main__':
    a = int(input())
    b = int(input())

    print(a//b)
    print(a/b)

# Introduction_Loops:

if __name__ == '__main__':
    n = int(input())
for i in range(n):  #range function finds the numbers from 0 to n
    print(i**2)

# Introduction_Write_a_function:

def is_leap(year):
    leap = False


    if year % 400 ==0:
       leap = True
    elif (year %4 ==0 and year % 100 !=0):
        leap = True
    return leap

# Introduction_Print_Function:

if __name__ == '__main__':
    n = int(input())

    for i in range(1,n+1):
        print(i, end='') # end='', helps to print with one line



#---------------------------------------------------------------

# BasicDataTypes_List_Comprehensions:

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())

coordinates = [[i, j, k] for i in range(x + 1) for j in range(y + 1) for k in range(z + 1) if i + j + k != n] #I write as x+1 because range should be from 0 to x
print(list(coordinates))

# BasicDataTypes_Find_theRunner-Up_Score!:

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
a=set(arr)
rs=sorted(a)
print(rs[-2])

# BasicDataTypes_Nested_Lists:
if __name__ == '__main__':

    alist=[]
    for _ in range(int(input())):
        name = input()
        score = float(input())
        alist.append([name,score])

    second_hscore = sorted(set([score for name, score in alist]))[1]
    names_with_second_hscore = sorted([name for name, score in alist if score == second_hscore])

    print('\n'.join(names_with_second_hscore))

# BasicDataTypes_Finding_thepercentage:

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()

    list1=list(student_marks[query_name])

    summary=sum(list1)
    average=summary/len(list1)
    print('%.2f'%  average)

# BasicDataTypes_Lists:

if __name__ == '__main__':
    N = int(input())
    list1=[]

    for _ in range(N):
        command = input().strip().split()

        if command[0] == 'insert':
            index = int(command[1])
            value = int(command[2])
            list1.insert(index, value)  # Insert value at the given index
        elif command[0] == 'print':
            print(list1)
        elif command[0] == 'remove':
            value = int(command[1])
            list1.remove(value)
        elif command[0] == 'append':
            value = int(command[1])
            list1.append(value)  # Append the element at the end of the list
        elif command[0] == 'sort':
            list1.sort()  # Sort the list
        elif command[0] == 'pop':
            list1.pop()  # Pop the last element from the list
        elif command[0] == 'reverse':
            list1.reverse()  # Reverse the list

# BasicDataTypes_Tuples:

if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())

    t=tuple(integer_list)
    print(hash(t))

#---------------------------------------------------------------

# Strings_sWAP_cASE:

def swap_case(s):
    string = ""

    for i in s:

        if i.isupper() == True:

            string+=(i.lower())

        else:

            string+=(i.upper())
    return string

# Strings_String_SplitandJoin:

def split_and_join(line):
    a=line.split(" ") #split words
    b="-".join(a) #put - between words
    return b

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)


# Strings_What's_Your_Name?:


def print_full_name(first, last):
    print("Hello " + first + " "+ last+ "!"+ " You just delved into python.")

# Strings_Mutations:

def mutate_string(string, position, character):
    list1=list(string)
    list1[position]=character
    string=''.join(list1)

    return string

# Strings_Find_a_string:

def count_substring(string, sub_string):
    result=0
    for i in range(len(string)):
        if string[i:len(string)].startswith(sub_string):
            result+=1

    return result

# Strings_String_Validators:

if __name__ == '__main__':
    s = str(input())

    print(any(i.isalnum() for i in s))
    print(any(i.isalpha() for i in s))
    print(any(i.isdigit() for i in s))
    print(any(i.islower() for i in s))
    print(any(i.isupper() for i in s))

# Strings_Text_Alignment:

tness=int(input())

var1='H'

for i in range(tness):
    print((var1*i).rjust(tness-1)+var1+(var1*i).ljust(tness-1))

for i in range(tness+1):
    print((var1*tness).center(tness*2)+(var1*tness).center(tness*6))

for i in range((tness+1)//2):
    print((var1*tness*5).center(tness*6))

for i in range(tness+1):
    print((var1*tness).center(tness*2)+(var1*tness).center(tness*6))

for i in range(tness):
    print(((var1*(tness-i-1)).rjust(tness)+var1+(var1*(tness-i-1)).ljust(tness)).rjust(tness*6))

# Strings_Text_Wrap:
import textwrap

def wrap(string, max_width):

    return textwrap.fill(string, max_width)

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)

# Strings_Designer_Door_Mat:

n,m = map(int, input().split())

for i in range(int(n/2)):
    st = ".|." * (2 * i + 1)
    x = st.center(m, '-')
    print(x)

print("WELCOME".center(m, '-'))

for i in reversed(range(int(n/2))):
    st = ".|." * (2 * i + 1)
    x = st.center(m, '-')
    print(x)

# Strings_String_Formatting:

def print_formatted(number):
    width = len('{0:b}'.format(number))

    for i in range(1, number + 1):
        print('{0:{width}d} {0:{width}o} {0:{width}X} {0:{width}b}'.format(i, width=width))

# Strings_Alphabet_Rangoli:

def print_rangoli(size):
    # your code goes here
    a = "abcdefghijklmnopqrstuvwxyz"
    data = [a[i] for i in range(size)]
    itm = list(range(size))
    itm = itm[:-1]+itm[::-1]
    for i in itm:
        temp = data[-(i+1):]
        row = temp[::-1]+temp[1:]
        print("-".join(row).center(n*4-3, "-"))

# Strings_Capitalize!:

def solve(s):
    result= s.split(' ')
    result1=[i.capitalize() for i in result ]
    return ' '.join(result1)

# Strings_Merge_theTools!:

import textwrap as tw
def merge_the_tools(string, k):
    # your code goes here
    for i in tw.wrap(string, k):
        seen = set()
        result = []

        for c in i:
            if c not in seen:
                seen.add(c)
                result.append(c)

        print(''.join(result))


#---------------------------------------------------------------

# Sets_Introduction_toSets:

def average(array):
    distinct_array=set(array)
    average= "{:.3f}".format(sum(distinct_array)/len(distinct_array))
    return average

# Sets_Symmetric_Difference:

m= int(input())
m_set= set(map(int,input().split()))
n= int(input())
n_set= set(map(int,input().split()))
a = (m_set.difference(n_set))
b = (n_set.difference(m_set))

ans = a.union(b)

for i in sorted(ans):
        print (i)

# Sets_Set.add():

n=int(input())
country=set()
for i in range(n):
    country.add(input())

print(len(country))

# Sets_Set.union()_Operation:

n=int(input()) #English newspaper subcribers
roll_n = set(map(int, input().split()))
b=int(input()) #French newspaper subcribers
roll_b = set(map(int, input().split()))

print(len(roll_n.union(roll_b)))

# Sets_Set.intersection()_Operation:

n=int(input()) #English newspaper subcribers
roll_n = set(map(int, input().split()))
b=int(input()) #French newspaper subcribers
roll_b = set(map(int, input().split()))

print(len(roll_n.intersection(roll_b)))

# Sets_Set.difference()_Operation:

n=int(input()) #English newspaper subcribers
roll_n = set(map(int, input().split()))
b=int(input()) #French newspaper subcribers
roll_b = set(map(int, input().split()))

print(len(roll_n.difference(roll_b)))

# Sets_Set.symmetric_difference()_Operation:

n=int(input()) #English newspaper subcribers
roll_n = set(map(int, input().split()))
b=int(input()) #French newspaper subcribers
roll_b = set(map(int, input().split()))

print(len(roll_n.symmetric_difference(roll_b)))

# Sets_Set_Mutations:

def updateit(set_A, s, command):
    if command=="update":
        set_A.update(s)
    elif command=="difference_update":
        set_A.difference_update(s)
    elif command=="intersection_update":
        set_A.intersection_update(s)
    elif command == "symmetric_difference_update":
        set_A.symmetric_difference_update(s)
    return set_A

A= int(input())
set_A= set(map(int, input().split()))

for i in range(int(input())):
    command, len_set=input().split()
    s= set(map(int, input().split()))
    set_A= updateit(set_A,s,command)

print(sum(set_A))

# Sets_TheCaptain's_Room:

k=int(input())
rooms=list(map(int,input().split()))
set_room= set(rooms)
sum_room=sum(rooms)
set_sum_rooms=sum(set_room)*k
captain_room=(set_sum_rooms- sum_room)//(k - 1)
print(captain_room)

# Sets_Check_Subset:

t = int(input())

for i in range(t):

    a= int(input())
    set_a= set(map(int,input().split()))

    b= int(input())
    set_b= set(map(int,input().split()))


    print(set_a.issubset(set_b))



#---------------------------------------------------------------

# Collections_collections.Counter():

from collections import Counter

shoes_number=int(input())
shoes_size= Counter(map(int, input().split()))
customer_number=int(input())

cost=0

for i in range(customer_number):
    size, price= map(int,input().split())
    if shoes_size[size]:
        cost += price
        shoes_size[size] -=1

print(cost)

# Collections_DefaultDict_Tutorial:

from collections import defaultdict
input_n, input_m = map(int, input().split())
d = defaultdict(list)

for i in range(input_n):
    res1=input()
    d[res1].append(i+1)
for j in range(input_m):
    res2=input()
    if res2 in d:
        print(*d[res2]) #* means unpack the list so that each line number is printed separately.

    else:
        print(-1)

# Collections_Collections.namedtuple():

from collections import namedtuple

n=int(input())
columns=input().split()

total_marks= 0

for i in range(n):
    students= namedtuple('my_student', columns)
    MARKS, CLASS, NAME, ID =input().split()
    my_student=students(MARKS, CLASS, NAME, ID)
    total_marks += int(my_student.MARKS)

print(total_marks/n)

# Collections_Collections.OrderedDict():

from collections import OrderedDict
liste=OrderedDict()
N=int(input())

for i in range(N):
    item_name,space , net_price= input().rpartition(' ') #rpartition instead of split() because item names can contain spaces, but the last part of the string will always be the price, separated by the last space.

    liste[item_name]= liste.get(item_name,0)+ int(net_price)

for item_name, net_price in liste.items():
    print(item_name, net_price)

#Collections_Word_Order

n=int(input())
counter={}
list_word=[]

for i in range(n):
    word=input()
    list_word.append(word)
    if word in counter:
        counter[word]+=1
    else:
        counter[word]=1
print(len(counter))

print(' '.join([str(counter[word]) for word in counter]))



# Collections_Collections.deque():

from collections import deque

d = deque()
for i in range(int(input())):
    command = input().split()
    if command[0] == 'append':
        d.append(command[1])
    elif command[0] == 'appendleft':
        d.appendleft(command[1])
    elif command[0] == 'pop':
        d.pop()
    else:
        d.popleft()
print(' '.join(d))

#Collections_Company_Logo

import math
import os
import random
import re
import sys


if __name__ == '__main__':
    s = input()
    s=sorted(s)

    frequency={}

    for i in s:
        if i in frequency:
            frequency[i] += 1
        else:
            frequency[i]=1

    sort_frequency=sorted(frequency.items(),key=lambda x:(-x[1],x[0]))
    for k,v in sort_frequency[:3]:
        print(k,v)

#Collections_Pilling_Up!

def check_cubes(cubes):
    left = 0
    right = len(cubes) - 1
    current_top = float('inf')

    while left <= right:
        if cubes[left] >= cubes[right]:
            selected_cube = cubes[left]
            left += 1
        else:
            selected_cube = cubes[right]
            right -= 1

        if selected_cube > current_top:
            return "No"
        current_top = selected_cube

    return "Yes"

r = int(input())
for _ in range(r):
    n = int(input())
    cubes = list(map(int, input().split()))
    print(check_cubes(cubes))

#---------------------------------------------------------------

# DateandTime_Calendar_Module:

import calendar

month, day, year= list(map(int,input().split()))

find=calendar.weekday(year,month,day)

print((calendar.day_name)[find].upper())

# DateandTime_Time_Delta

import math
import os
import random
import re
import sys
from datetime import  datetime,timedelta
# Complete the time_delta function below.
def time_delta(t1, t2):
    date_format= "%a %d %b %Y %H:%M:%S %z"
    time1=datetime.strptime(t1,date_format)
    time2=datetime.strptime(t2,date_format)
    time_dif= abs(int((time1-time2).total_seconds()))
    return str(time_dif)
    
    
if __name__ == '__main__':
    #fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)
        print(delta)
        #fptr.write(delta + '\n')

    #fptr.close()


#---------------------------------------------------------------

# ErrorsandExceptions_Exceptions:

for i in range(int(input())):
    try:
        a, b = map(int, input().split())
        print(int(a//b))

    except Exception as ex:
        print("Error Code:",ex)

#---------------------------------------------------------------

# Built-ins_Zipped!:

n,x= map(int, input().split())

a =[]

for i in range(x):
    a.append(map(float,input().split()))

for i in zip(*a):
    print(sum(i)/len(i))

# Built-ins_ginortS:

def srt_string(s):
    lower_case = []
    upper_case = []
    odd_digits = []
    even_digits = []
    for char in s:
        if char.islower():
            lower_case.append(char)
        elif char.isupper():
            upper_case.append(char)
        elif char.isdigit():
            if int(char)%2==0:
                even_digits.append(char)
            else:
                odd_digits.append(char)
    return (''.join(sorted(lower_case)+sorted(upper_case)+sorted(odd_digits)+sorted(even_digits)))

if __name__=='__main__':

    S= input()
    print(srt_string(S))

#Built-Ins_Athlete_Sort

import math
import os
import random
import re
import sys
from operator import itemgetter


if __name__ == '__main__':
    first_multiple_input = input().rstrip().split()

    n = int(first_multiple_input[0])

    m = int(first_multiple_input[1])


    rows = []
    for _ in range(n):
        rows.append(list(map(int, input().rstrip().split())))


    k = int(input().strip())


    arr = sorted(rows, key=itemgetter(k))


    for row in arr:
        print(*row)

#---------------------------------------------------------------

# PythonFunctionals_Map_and_LambdaFunction:

cube = lambda x: pow(x,3) # complete the lambda function

def fibonacci(n):
    # return a list of fibonacci numbers
    list1= [0,1]
    for i in range(2,n):
        list1.append(list1[i-2]+list1[i-1])
    return list1[0:n]

#---------------------------------------------------------------

# RegexandParsing_Detect_Floating_Point_Number:

import re

n = int(input())
for i in range(n):
    t = input()
    print (bool(re.match(r"^[-+]?[0-9]*\.[0-9]+$",t)))

# RegexandParsing_Re.split():

regex_pattern = r""	# Do not delete 'r'.


regex_pattern = r"[.]|[,]"

# RegexandParsing_Group(),Groups()&Groupdict():

import re

match = re.search(r'([a-zA-Z0-9])\1+', input().strip())

print(match.group(1) if match else -1)

# RegexandParsing_Re.findall()&Re.finditer():

import re

s= input()
v= "aeiou"
c="qwrtypsdfghjklzxcvbnm"

l=re.findall(r"(?<=[%s])([%s]{2,})[%s]" % (c,v,c),s, flags=re.IGNORECASE)

if not l:
    print(-1)
else:
    for i in l:
        print(i)

# RegexandParsing_Re.start()&Re.end():

import re


a = input().strip()
b = input().strip()

found = False
n = len(a)
m = len(b)

for i in range(n):
    if a[i:i + m] == b:     # substring starting at index `i` matches `b`
        print((i, i + m - 1))
        found = True

if not found:
    print((-1, -1)) # no match is found, print (-1, -1)

# RegexandParsing_Validating_phone_numbers:

import re
n=int(input())

for i in range(0,n):
    print("YES") if re.match(r'[789]\d{9}$', input())else print("NO")

# RegexandParsing_Validating_and_Parsing_EmailAddresses:

import re
import email.utils

n = int(input())

pattern = r'^[a-z][\w\-\.]+@[a-z]+\.[a-z]{1,3}$'

for i in range(0, n):
    p_addr = email.utils.parseaddr(input())
    if re.search(pattern, p_addr[1]):
        print(email.utils.formataddr(p_addr))


# RegexandParsing_Hex_Color_Code:

import re

for i in range(int(input())):

    match = re.findall(r"(\#[a-f0-9]{3,6})[\;\,\)]", input(), re.I)
    for color in match:
        print(color)

# RegexandParsing_HTML_Parser-Part1:

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attributes):
        print('Start :', tag)
        for attr in attributes:
            print('->', attr[0], '>', attr[1])

    def handle_endtag(self, tag):
        print('End   :', tag)

    def handle_startendtag(self, tag, attributes):
        print('Empty :', tag)
        for attr in attributes:
            print('->', attr[0], '>', attr[1])

N = int(input())
Parser = MyHTMLParser()
Parser.feed(''.join(input().strip() for _ in range(N)))
#Note: I tried to write the general outline of the code myself, but there were places where I revised it by looking at examples on github

# RegexandParsing_HTML_Parser-Part2:

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if len(data.split('\n')) != 1:
            print(">>> Multi-line Comment")
        else:
            print(">>> Single-line Comment")
        print(data.replace("\r", "\n"))

    def handle_data(self, data):
        if data.strip():
            print(">>> Data")
            print(data)

html = ""
for i in range(int(input())):  # Use input() for Python 3
    html += input().rstrip() + "\n"

parser = MyHTMLParser()
parser.feed(html)
#Note: I tried to write the general outline of the code myself, but there were places where I revised it by looking at examples on github

# RegexandParsing_Detect_HTML_Tags,Attributes_and_Attribute Values:

from html.parser import HTMLParser

N = int(input())

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attributes):
        print(tag)
        for attribute in attributes:
            print('-> {} > {}'.format(*attribute))

html = '\n'.join(input() for _ in range(N))
parser = MyHTMLParser()
parser.feed(html)
#Note: I tried to write the general outline of the code myself, but there were places where I revised it by comparing it with some of the solutions on github

# RegexandParsing_Validating UID:

import re

if __name__ == "__main__":
    T = int(input().strip())

    for _ in range(T):
        uid = "".join(sorted(input()))
        if (len(uid) == 10 and re.match(r'', uid) and re.search(r'[A-Z]{2}', uid) and re.search(r'\d\d\d', uid) and not re.search(r'[^a-zA-Z0-9]', uid) and not re.search(r'(.)\1', uid)):
            print("Valid")
        else:
            print("Invalid")

# RegexandParsing_Validating_Credit_Card_Numbers:

def check(t):
    if t[0]!='4' and t[0]!='5' and t[0]!='6':     #start with a 4,5 or 6
        return False

    if len(t.replace("-",""))!=16 :     #contain exactly 16 digits.
        return False

    for i in range(len(t)): #only consist of digits (0-9).
        if t[i] == '-':
            continue
        if not t[i].isdigit():
            return False

    d=0    #have digits in groups of 4, separated by one hyphen "-".

    for i in range(len(t)):
        if (t[i]=='-' and d!=4):
            return False
        elif  ((t[i])=='-' and d==4):
            d=0
        else:
            d+=1

    if (" " in t or "_" in t): #NOT use any other separator like ' ' , '_', etc.
        return False

    c=0 #NOT have 4 or more consecutive repeated digits.
    cc=t.replace("-","")
    for i in range(len(cc)-1):
        if(cc[i]!=cc[i+1]):
            c=0
        else:
            c+=1
        if c==3:
            return False
    return True

num=input()

for i in range(int(num)):
    t=input()
    if (check(t)):
        print("Valid")

    else:
        print("Invalid")

#---------------------------------------------------------------

# XML_XML1-Find_theScore:

import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    return len(node.attrib) + sum(get_attr_number(child) for child in node)

if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))

# XML_XML2-Find_theMaximum_Depth:
import xml.etree.ElementTree as etree

maxdepth = 0
def depth(elem, level):
    global maxdepth
    if (level == maxdepth):
        maxdepth += 1

    for child in elem:
        depth(child, level + 1)
if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)

#---------------------------------------------------------------

# ClosuresandDecorators_Standardize_Mobile_Number_Using_Decorators:

def wrapper(f):
    def fun(l):
        f(["+91 "+c[-10:-5]+" "+c[-5:] for c in l])
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l)

# ClosuresandDecorators_Decorators2-Name_Directory:

import operator

def person_lister(f):
    def inner(people):
        return [f(p) for p in sorted(people, key=lambda x: int(x[2]))]

    return inner

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')

#---------------------------------------------------------------

# Numpy_Arrays:

import numpy


def arrays(arr):
    return(numpy.array(arr[::-1], float))

arr = input().strip().split(' ')
result = arrays(arr)
print(result)

# Numpy_ShapeandReshape:

import numpy as np
n= np.array(list(map(int,input().split())))

n.shape=(3,3)

print (n)

# Numpy_Transpose_and_Flatten:

import numpy as np

n, m= input().split()

ln=[]

for i in range(int(n)):
    ln.append(list((map(int,input().split()))))

my_arr= np.array(ln)

print (np.transpose(my_arr))
print (my_arr.flatten())

# Numpy_Concatenate:

import numpy as np

elements=input().split()
n = int(elements[0])
m = int(elements[1])
p = int(elements[2])

l=[]

for i in range(n):
    n_row= list(map(int,input().split()))
    l.append(n_row)

mt=[]
for i in range(m):
    m_row= list(map(int,input().split()))
    mt.append(m_row)
l_array = np.array(l)
mt_array = np.array(mt)
result = np.concatenate((l_array, mt_array), axis=0)


print(result)

# Numpy_Zeros_and_Ones:

import numpy as np

def arrays(arr):
    arr1=  np.array(arr,int)
    arr2= np.zeros(arr1,int)
    arr3=np.ones(arr1,int)
    return arr2,arr3

arr = input().strip().split(' ')
arr2, arr3 = arrays(arr)

print(arr2)
print(arr3)

# Numpy_Eye_and_Identity:

import numpy as np
np.set_printoptions(sign=' ')

n,m = map(int, input().split())
eye = np.eye(n,m)
print(eye)

# Numpy_Array_Mathematics:

import numpy as np

def arrays(arr_a, arr_b):
    arr_a = np.array(arr_a, dtype=int)
    arr_b = np.array(arr_b, dtype=int)

    add = np.add(arr_a, arr_b)
    sub = np.subtract(arr_a, arr_b)
    mul = np.multiply(arr_a, arr_b)
    div = np.floor_divide(arr_a, arr_b)
    mod = np.mod(arr_a, arr_b)
    power = np.power(arr_a, arr_b)

    return add, sub, mul, div, mod, power

n, m = map(int, input().split())

arr_a = [list(map(int, input().split())) for _ in range(n)]

arr_b = [list(map(int, input().split())) for _ in range(n)]


add, sub, mul, div, mod, power = arrays(arr_a, arr_b)

print(add)
print(sub)
print(mul)
print(div)
print(mod)
print(power)

# Numpy_Floor,CeilandRint:

import numpy as np

np.set_printoptions(sign=' ')

def arrays(arr):
    arr_a=np.array(arr,dtype=float)
    floor=np.floor(arr_a)
    ceil=np.ceil(arr_a)
    rint=np.rint(arr_a)

    return floor,ceil,rint

arr=input().strip().split(' ')

floor, ceil, rint = arrays(arr)
print(floor)
print(ceil)
print(rint)

# Numpy_Sum_and_Prod:

import numpy as np

n,m=map(int,input().split())

arr=np.array([input().split() for i in range(n)],int)

print(np.prod(np.sum(arr,axis=0),axis= None))

# Numpy_Min_and_Max:

import numpy as np

n,m = map(int, input().split())

array = np.array([input().split() for i in range(n)], int)

print  (np.max(np.min(array, axis=1),axis=0))

# Numpy_Mean,Var,andStd:

import numpy as np
N,M = map(int, input().split(" "))
A = np.array([input().split() for i in range(N)],int)
print(np.mean(A, axis = 1))
print(np.var(A, axis = 0))
print(round(np.std(A, axis = None),11))

# Numpy_Dot_and_Cross:

import numpy as np

n=int(input())

a=np.array([input().split() for i in range(n)], int)

b=np.array([input().split() for i in range(n)], int)
c=np.dot(a,b)

print(c)

# Numpy_Inner_and_Outer:

import numpy as np

a = np.array(input().split() , int)
b = np.array(input().split() , int)
print(np.inner(a, b))
print(np.outer(a, b))

# Numpy_Polynomials:

import numpy as np

m=np.array(input().split(), float)
n= float(input())

print(np.polyval(m,n))

# Numpy_Linear_Algebra:

import numpy as np

np.set_printoptions(legacy='1.13')

n = int(input())

A = np.array([input().split() for i in range(n)], float)
print(np.linalg.det(A))


#**********PROBLEM 2**********

#Birthday Cake Candles:


import math
import os
import random
import re
import sys


def birthdayCakeCandles(candles):
    m=max(candles)
    return candles.count(m)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

#Number Line Jumps:

import math
import os
import random
import re
import sys


def kangaroo(x1, v1, x2, v2):
    if v1 == v2:     # If both have the same velocity and start at different positions

        return "YES" if x1 == x2 else "NO"

    # If one kangaroo is behind and jumps equal or less distance
    if (x1 < x2 and v1 <= v2) or (x2 < x1 and v2 <= v1):
        return "NO"

    # Check if they will meet based on positions and velocities
    # Ensure we check which kangaroo is behind
    if v1 > v2 and x1 < x2:
        return "YES" if (x2 - x1) % (v1 - v2) == 0 else "NO"
    elif v2 > v1 and x2 < x1:
        return "YES" if (x1 - x2) % (v2 - v1) == 0 else "NO"

    return "NO"

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

#Viral Advertising:


import math
import os
import random
import re
import sys

def viralAdvertising(n):
    shared = 5
    cumulative = 0
    for i in range(1,n+1):
        liked = shared//2
        cumulative+=liked
        shared = liked*3
    return cumulative


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

#Recursive_Digit_Sum

import math
import os
import random
import re
import sys


def superDigit(n, k):
    def get_super_digit(num):
        if len(num) == 1:
            return int(num)
        else:
            return get_super_digit(str(sum(int(digit) for digit in num)))

    initial_sum = sum(int(digit) for digit in n) * k

    return get_super_digit(str(initial_sum))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

#Insertion Sort - Part 1:


import math
import os
import random
import re
import sys


def insertionSort1(n, arr):
    point=arr[-1]
    i=n-1
    while i >0 and arr[i-1]> point:
        arr[i]=arr[i-1]
        print(*arr)
        i -=1
    arr[i]=point
    print(*arr)


if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

#Insertion Sort - Part 2:


import math
import os
import random
import re
import sys

def insertionSort2(n, arr):
    for k in range(1, n):
        point=arr[k]
        t = k
        while t >0 and arr[t-1]> point:
            arr[t]=arr[t-1]
            t -=1
        arr[t]=point
        print(*arr)



if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)
