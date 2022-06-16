# Neural_Networks_Implementations
Implementations of MLP and RNN architectures, including different optimizers and loss functions in Python.


### User Guide for naive_bayes_spam_detector.ipynb

Functions:<br/>
  * load_dataset(x_filename, y_filename): This method takes two file paths; "x_train.csv" and "y_train.csv" 
  if we assume the csv files are in the same folder as the q3main.py file, and returns one 2D numpy array
  train_x, one 1D numpy array train_y and set_size (number of rows).<br/>

  * naive_bayes_train(train_x, train_y, set_size, alpha, mult): alpha is the Dirichlet prior used in part 3.3.
  For the other parts use alpha = 0. mult is a boolean which means multinomial. Use True for parts 3.2 and 3.2
  Use False for part 3.4.
  Return values are: spam_ratio = P(Y = spam), normal_ratio = P(Y = normal), 
  spam_occurence_ratios = P(Xj | Y = spam) and normal_occurence_ratios = P(Xj | Y = normal)
  definition of spam_occurence_ratios and normal_occurence_ratios base on the type of model used; 
  multinomial or binomial or if alpha is being used.<br/>

  * naive_bayes_test(test_x, test_y, spam_ratio, normal_ratio, spam_occurence_ratios, normal_occurence_ratios, mult):
  The parameters are same as parameters and return values of naive_bayes_train. Return values are:
  results: a matrix of size set_size with 1s for spam mails and 0s for normal mails. no_of_wrong is the number of 
  wrong estimations, accuracy is not a percentage; it's a float number between 0 and 1.<br/>
  
  * data.zip includes: x_train.csv, y_train.csv, x_test.csv, y_test.csv, vocabulary.txt
  x_train.csv has one row for each mail, each column corresponds to one word in the vocabulary. x_train[i,j] corresponds
  to the number of occurences of the word j (jth word in the vocabulary) in the ith mail in the training data.
  y_train.csv also has one row for each mail and it has one column. If the value in column i is 1 then that means
  the ith mail in the training set is spam. x_test.csv and y_test.csv have the same logic as x_train.csv and y_train.csv.
  Vocabulary.txt stores all the words used in the mails.<br/> 
 
  Numbers, punctuation, words which are used the most in English such as a, the and the words which appear only once
  in the whole data were removed in data preprocessing.
