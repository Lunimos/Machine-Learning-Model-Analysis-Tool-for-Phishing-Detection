import pandas as pd
import time
from scipy.io import arff
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# define global variables
tree_classifier = DecisionTreeClassifier
random_forest_classifier = RandomForestClassifier

def wrong_data():
    '''Function for wrong inputs'''
    print("""
    ----------------------------
    | Incorrect Option         |
    |                          |
    | Please Try Again         |
    |                          |
    | Watch For Typo's         |
    ----------------------------
    """)
def ML_algo(training_inputs, training_outputs, testing_inputs, testing_outputs):
    '''Main menu to choose a machine learning algorithm'''
    global tree_classifier, random_forest_classifier
    print("""
    -------------------
    | Algorithm Menu   |       
    -------------------
           """)

    print("""
    Press 1 for Decision Tree
    Press 2 for Random Forest
    or B to go back 
    Please Select An Option""")
    
    userinput = input(" ")
    if userinput == "1":
        tree = DecisionTreeClassifier()
        tree.fit(training_inputs, training_outputs)
        predictions = tree.predict(testing_inputs)
        accuracy = 100 * accuracy_score(testing_outputs, predictions)
        precision = 100 * precision_score(testing_outputs, predictions, pos_label='1')
        recall = 100 * recall_score(testing_outputs, predictions, pos_label='1')
        f1 = 100 * f1_score(testing_outputs, predictions, pos_label='1')
        print("Decision Tree:")
        print("Accuracy: {:.2f}%".format(accuracy))
        print("Precision: {:.2f}%".format(precision))
        print("Recall: {:.2f}%".format(recall))
        print("F1 Score: {:.2f}%".format(f1))

        
    elif userinput == "2":
        randomforest = RandomForestClassifier()
        randomforest.fit(training_inputs, training_outputs)
        predictions = randomforest.predict(testing_inputs)
        accuracy = 100 * accuracy_score(testing_outputs, predictions)
        precision = 100 * precision_score(testing_outputs, predictions, pos_label='1')
        recall = 100 * recall_score(testing_outputs, predictions, pos_label='1')
        f1 = 100 * f1_score(testing_outputs, predictions, pos_label='1')
        print("Random Forest:")
        print("Accuracy: {:.2f}%".format(accuracy))
        print("Precision: {:.2f}%".format(precision))
        print("Recall: {:.2f}%".format(recall))
        print("F1 Score: {:.2f}%".format(f1))


    elif userinput.lower() == "b":
        return None  # Indicate to go back
    else:
        wrong_data()
        time.sleep(2)
        return ML_algo(training_inputs, training_outputs, testing_inputs, testing_outputs)  # Retry algorithm choice

def load_user_data(return_to_menu):
    print("""
          --------------
         | DATA LOADER  |       
          -------------- 
           """)
    
    while True:
        print("Only ARFF and CSV file types supported.")
        print("Press Q to go back.")
        dataset_name = input("Please enter the file name with the extension: ")
        
        if dataset_name.lower() == "q":
            # Go back to the previous menu
            return_to_menu ()

        try:
            # Check if the file has an ARFF extension
            if dataset_name.endswith('.arff'):
            # Load ARFF data
                data, meta = arff.loadarff(dataset_name)
                df = pd.DataFrame(data)
                # Convert bytes to string (if necessary)
                df = df.map(lambda x: x.decode() if isinstance(x, bytes) else x)
                print(f"Data has been loaded successfully from '{dataset_name}' (ARFF)")

            elif dataset_name.endswith('.csv'):
                df = pd.read_csv(dataset_name)
                print(f"Data has been loaded successfully from '{dataset_name}' (CSV)")
            else:
                print("Unsupported file format.")
                continue
                
            data_response = input("Do you want to use this data? (y/n): ")
            if data_response.lower() == "y":
                time.sleep(2)
                print(f"Using '{dataset_name}'")
                inputs = df.iloc[:, :-1]
                outputs = df.iloc[:, -1]
                training_size = 2000
                training_inputs = inputs[:training_size]
                training_outputs = outputs[:training_size]
                testing_inputs = inputs[training_size:]
                testing_outputs = outputs[training_size:]
                
                classifier = ML_algo(training_inputs, training_outputs, testing_inputs, testing_outputs)
                if classifier is not None:
                    return training_inputs, training_outputs, testing_inputs, testing_outputs, classifier
                else:
                    continue
                
            elif data_response.lower() == "n":
                print("Choose another dataset.")
                continue
            else:
                print("Invalid response. Please enter 'y' or 'n'.")
                
        except FileNotFoundError:
            print("File not found.")
            continue
        except IOError:
            print("Error reading file.")
            continue
        except Exception as e:
            print("An error occurred:", e)
            continue

if __name__ == "__main__":
    training_inputs, training_outputs, testing_inputs, testing_outputs, classifier = load_user_data()
    if training_inputs is not None:
        # Do something with the data and classifier, e.g., train and evaluate the model
        print("Data and classifier loaded successfully!")