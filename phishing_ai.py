import sys
import time
from tester import load_user_data
def wrong_data():
    '''Function for wrong inputs'''
    # Function for showing some data was wrong
    print("""
    ----------------------------
    | Incorrect Option         |
    |                          |
    | Please Try Again         |
    |                          |
    | Watch For Typo's         |
    ----------------------------
    """)

def usermenu():
    '''Main menu to analyze a file or quit'''
    # Print statement to show we are in the main menu
    print("""
          --------------
         | MAIN MENU    |       
          -------------- 
           """)

    print("""
    Press A To Train the AI
    Press B To Analyze A Website
    Press Q To Quit The Program
    Please Select An Option""")
    # Set the user input to empty
    userinput = input(" ")
    if userinput == "A" or userinput == "a":
        load_user_data(return_to_menu=usermenu)
       # file_path = input("Enter the path to the DataSet you want to analyze: ")
      

    elif userinput == "B" or userinput == "b":
        user_input = input("Please Enter The Website")
       

    elif userinput == "Q" or userinput == "q":
        print("Quitting the program.")
        

    else:
        wrong_data() 
         # Call the wrong_data function for incorrect options
        time.sleep(2)
        main()

def main():
    '''Main function for calling the first menu'''
    # Call the main menu
    usermenu()

if __name__ == "__main__":
    main()