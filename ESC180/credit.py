
"""The Credit Card Simulator starter code
You should complete every incomplete function,
and add more functions and variables as needed.
Ad comments as required.
Note that incomplete functions have 'pass' as the first statement:
pass is a Python keyword; it is a statement that does nothing.
This is a placeholder that you should remove once you modify the function.
Author: Michael Guerzhoy.  Last modified: Oct. 3, 2022
"""

# You should modify initialize()
def initialize():
    global cur_balance_owing_intst, cur_balance_owing_recent, disabled
    global last_update_day, last_update_month
    global last_country, last_country2
    
    cur_balance_owing_intst = 0
    cur_balance_owing_recent = 0
    
    last_update_day, last_update_month = -1, -1
    
    last_country = None
    last_country2 = None    

    disabled = False    

def date_same_or_later(day1, month1, day2, month2):
    return day1 == day2 and month1 == month2 or day1 > day2 and month1 == month2 or month1 > month2

def all_three_different(c1, c2, c3):
    return c1 != c2 != c3 != c1 

def purchase(amount, day, month, country):
    global last_country, last_country2, last_update_day, last_update_month, cur_balance_owing_intst, cur_balance_owing_recent, disabled
    if disabled:
        return 'error'
    if country != last_country != last_country2 != country and last_country2 != None:
        disabled = True
        return 'error'
    
    if not date_same_or_later(day, month, last_update_day, last_update_month):
        return 'error'
    
    interest(month, last_update_month)

    if month == last_update_month:
        cur_balance_owing_recent += amount
    
    else:
        cur_balance_owing_recent += amount
    
    last_country2, last_country = last_country, country
    last_update_day, last_update_month = day, month 

def amount_owed(day, month):
    global last_country, last_country2, last_update_day, last_update_month, cur_balance_owing_intst, cur_balance_owing_recent
    
    if not date_same_or_later(day, month, last_update_day, last_update_month):
        return 'error'
    #print(f'{cur_balance_owing_intst} = {cur_balance_owing_intst} * (1.05 ** ({month} - {last_update_month})) + {cur_balance_owing_recent} * (1.05 ** ({month} - {last_update_month} - 1))')
    #print(month, last_update_month)
    
    interest(month, last_update_month)

    #print(cur_balance_owing_intst, cur_balance_owing_recent)

    last_update_day, last_update_month = day, month 

    return cur_balance_owing_intst + cur_balance_owing_recent

def pay_bill(amount, day, month):
    global last_country, last_country2, last_update_day, last_update_month, cur_balance_owing_intst, cur_balance_owing_recent
    
    if not date_same_or_later(day, month, last_update_day, last_update_month):
        return 'error'
    #print(cur_balance_owing_intst, cur_balance_owing_recent)
    interest(month, last_update_month)
    if month != last_update_month:
        cur_balance_owing_intst += cur_balance_owing_recent
        cur_balance_owing_recent = 0

    day, month = last_update_day, last_update_month
    
    if cur_balance_owing_intst != 0 and cur_balance_owing_intst > amount:
        cur_balance_owing_intst -= amount
    
    else:
        amount -= cur_balance_owing_intst
        cur_balance_owing_intst = 0
        cur_balance_owing_recent -= amount
    #print(cur_balance_owing_intst, cur_balance_owing_recent)

def interest(month, last_month):
    global last_country, last_country2, last_update_day, last_update_month, cur_balance_owing_intst, cur_balance_owing_recent
    if month - last_update_month  == 1 and month != 2:
        cur_balance_owing_intst = cur_balance_owing_recent + cur_balance_owing_intst *1.05
        cur_balance_owing_recent = 0
    elif month != last_month and month - last_update_month != 1:
        cur_balance_owing_intst = cur_balance_owing_intst * (1.05 ** (month - last_update_month)) + cur_balance_owing_recent * (1.05 ** (month - last_update_month - 1))
        cur_balance_owing_recent = 0

initialize()
