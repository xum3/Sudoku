import csv


# main function
def solver(input):
	pass
	#return output


# below is support functions --------------------------------------------------------

# intro: check accuracy by compare two dictionary
# args: 
#	- answer : output dict returns by solver(input)
#	- solution: correct solution dict
# return: a float number that indicate how acurate the answer dict is (e.g. 0.89)
def checkAccuracy(answer,solution):
	correct = 0
	wrong = 0
	for key in answer.keys():
		if (answer[key] == solution[key]):
			correct = correct +1
		else:
			wrong = wrong +1

	return correct/(correct + wrong)





# above is support functions --------------------------------------------------------






# below code will be executed when the program runs
if __name__ == "__main__":

	# Define global variables below -------------------------------
	solutionDict = dict()
	mySolutionDict = dict()

	# Define global variables above -------------------------------


	with open('TestSet/small1.csv', newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		i = 1
		for row in reader:
			print(row['quizzes'])
			mySolutionDict[i] = row['quizzes']
			solutionDict[i] = row['solutions']
			i = i + 1

	print(checkAccuracy(mySolutionDict,solutionDict))