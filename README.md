# Character-Level-LSTM-for-Parameter-Injection-Detection
Web application attacks are one of the main points of attack against web servers. With the increase in exchange of information through the web, attackers find new ways to compromise the security of the web and their users. HTTP is an application layer stateless protocol that is widely used for communication between web browsers and we servers. Attackers often inject malicious codes into the parameters of the HTTP requests to execute their attacks against web servers. An automated approach for detecting web application attacks is proposed. By using the concept of Long Short-Term Memory, HTTP request packets will be inspected and analyzed for anomaly detection.



# HTTP DATASET CSIC 2010 (http://www.isi.csic.es/dataset/)
The HTTP dataset CSIC 2010 is used in this project. The CSIC 2010 contains thousands of web requests automatically generated by producing traffic to an e-Commerce web application. The dataset contains 36,000 HTTP requests labeled normal and about 25,000 requests labeled anomalous. The type of attacks includes, SQL injection, buffer overflow, information gathering, XSS and so on. The dataset was be preprocessed according to the neural network model. Some of the features included in the HTTP request which does not assist in distinguishing between a normal request and an anomalous one was removed, features such as protocol, userAgent, pragma, cacheControl, accept, acceptEncoding, acceptCharset, acceptLanguage, connection.
check out the following repository for HTTP CSIC 2010 preprocessing: 
https://github.com/Monkey-D-Groot/Machine-Learning-on-CSIC-2010/blob/master/main.py

# Result: 
The model was trained on partial training data and evaluated on the evaluation set over 100 epochs. After analyzing the results, it was determined that the validation accuracy stops improving after 40 epochs and the loss increases. Therefore, the number of epochs was set to 40. The model was rebuilt and trained over the entire data set and evaluated using the testing data which resulted in accuracy of %98. 
