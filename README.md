
Malware classiﬁcation on API traces
===================================

Assingment desctiption
----------------------

In this project, you will work on a unique dataset comprising approximately 50,000 recent traces of Windows malware. Each sample is annotated with the hash of the executable ﬁle.

Your task involves using annotations that are provided, which include ﬁnegrained results of Yara rules. In these annotations – these are JSON ﬁles mapping tags to arrays of hashs – each sample may have multiple tags, with a total of 187 possible tags. (You may later decide to group multiple tags into labels, or ignore some tags).

The tasks are as follows:
 1. Data Preprocessing: The dataset is structured with one ﬁle per sample, containing one line per API call. Extract the API call names from the traces to prepare the data for analysis. (You may collaborate with the second group on this task to preprocess the dataset.)
 2. Data Splitting: Divide the dataset into training, validation, and test subsets. Use the validation subset to ﬁnd the “best” approach, and use the test subset only for the ﬁnal
evaluation.
 3. Machine Learning Classiﬁer: Your primary goal is to develop and train a machine
 learning classiﬁer that can predict labels for API call traces based on the data and annotations available.

In [A3](https://reykjavik.instructure.com/courses/7360/assignments/72060), it turned out that the best solutions generally used a simple bag-of-words (or rather bag-of-API-calls) approach with a random forest classiﬁer. An excellent project is expected to make another attempt to explore appraoches that consider the order of /  sequences of API calls as a basis to improve classiﬁcation.

dataset
-------
Preprocessed dataset containing a list of API calls in a .txt file per virus execution available at this [google drive](https://drive.google.com/drive/folders/1MIeGFoPN31bOtrMAiAi8hC3sneU95H9J?usp=sharing).
Size of the dataset is 2.1GB.


