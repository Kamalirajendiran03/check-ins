PRIVACY-PRESERVING LOCATION CHECK-IN CATEGORY PREDICTION USING FEDERATED LEARNING
Location-Based Services (LBS) play a critical role in modern digital applications, offering functionalities ranging from navigation to social recommendations. However, they simultaneously pose serious privacy threats by collecting and storing sensitive user location data. Sharing raw check-in data with central servers opens vulnerabilities that can result in misuse of user movement patterns and preferences, violating privacy regulations like GDPR.
To address this, our project integrates two major innovations:
Web-Based Data Collection: A secure Flask application captures user check-in data locally into a SQLite database.
Privacy-Preserving Federated Learning Framework: Using the Flower framework, model training occurs locally on user devices without sharing raw data, preserving data privacy.
We use two real-world datasets, Foursquare NYC & Tokyo Check-ins and Gowalla Check-ins, involving millions of location records. Preprocessing includes missing data handling, feature extraction from timestamps, and normalization. Our integrated system achieves effective venue category prediction while ensuring that raw user data remains secure and decentralized.

INTRODUCTION
With the growing dependence on Location-Based Services, safeguarding the privacy of user check-in data has become an urgent priority. Traditional centralized machine learning approaches require users to upload their complete datasets to a server, risking massive data exposure in the event of a breach. In contrast, Federated Learning (FL) enables decentralized training, allowing users to keep their data locally while contributing to a global model collaboratively.
This project focuses on applying federated learning to the problem of venue category prediction based on user check-ins. Our goal is to achieve a strong predictive performance while maintaining strict compliance with privacy regulations such as GDPR. By integrating FL with a web-based data collection interface, we offer an end-to-end solution for privacy-conscious model training in location prediction applications.

SYSTEM OVERVIEW

Our proposed system architecture includes:
Flask Web Application: Users submit check-in details through an HTML form. The application stores submitted data into a local SQLite database (checkin.db) with automated table creation.
Federated Learning Framework (Flower): Clients train local models on their own check-in datasets. Model weight updates, rather than raw data, are securely sent to a central server.
Secure Aggregation Server: The server aggregates multiple client updates using Federated Averaging (FedAvg) without accessing individual updates, then sends the improved global model back to the clients.
This decentralized approach minimizes the risk of data leakage, as raw user data never leaves their local device.
