


Project statement:  To develop a model which predicts the virality of an article in the shortest minimum time after it is published on the website.


Project scope: This project is only developing virality predictive algorithm for text articles getting published on ww.punjabkesari.in. 
Articles getting published on android and ios apps is out of scope for this project. 

Project owner: Dhruv Patel 

Project Team: Ravi Pathak, Jigal Navadiya, Viraj Shah, Surender, Rajiv Pathak, Nitya Prakash.



Project timeline: 


Project Methodology: 

Data Sources :

For This project two data sources are being used :

GA360 data for punjab kesari
CMS data of articles provided by punjab kesari 

Data from December,2018 to May,2019 is used for building the model.
GA Data :

Following parameters for every article has been derived from google analytics.

Page  : A page/article on the website specified by path and/or query parameters. Use this with hostname to get the page's full URL.

Hour of Day : It shows the date and time in the format of YYYYMMDDHH when the pageview hit was received.

Pageviews  : The total number of pageviews for the particular page for that hour.(As per hour of the day parameter)

Unique Pageviews : Unique Pageviews is the number of sessions during which the specified page was viewed at least once. A unique pageview is counted for each page URL + page title combination for that hour.(of hour of the day parameter)

CMS data:

Article Content: All the textual content (in hindi) of article was provided in xml format by client.

Date of Article : This indicates publishing date and time of the article. Its format is DD/MM/YYYY  HH:MM:SS AM/PM.

Head line: It is the headline of the article when it was published on the website.

Story tags: Tags given to the article on website.

Key:
Article ID is the key parameter to merge two datasets.

Approach:

Our goal is to predict the number of unique page views the article will receive in the first 3 days of publishing. We are planning to make two types of models for predictions. One model will predict the potential page views before publishing the article just based on  content. The other model will make predictions past three hours of publication after analyzing response received by article in first 3 hours. 

-For this first step is to prepare a data set containing below features:
Independent Features:
-Features based on topic modelling from article content(15 topics)
-sentiment polarity of the article from article content
-Subjectivity of article from content
-Number of persons,locations, entities etc. mentioned in article
-Article Category
-Publishing day of article
-Publishing time of article
-Average unique page views of the 10 most similar articles in last month
-Unique page views of  article in first 3 hours

Dependent Feature:
-Unique Page views of article in first 3 days



Instructions for replicating the project for other client:
Step1:
-First get the content of the article along with his unique article_id and publishing date either by scrapping client website or  directly from client CRM data. 
-Estimate the number of articles published by client in a month(from GA or their website). That will give idea about timeline of articles to be used in training. For our purpose try use minimum 1 lakhs articles(define your timeline accordingly)
-Convert articles in english using Google Translate API/ or any other convenient way if they are in any regional language as most of the APIs are developed for english language only.

Step2:
-Once your article content is ready, collect data for that timeline from GA or Bigquery. Remember in GA you will get  most of the required columns directly(article wise as we need) but in Bigquery, you will have to convert hit-level data into required form.
-From GA, extract the unique page views data(hourwise data) for each article in decided data range keyed by their page url. Clean up irrelevant page urls i.e if you are building model for website only than remove app traffic identified by some keywords in pageurl. Extract news_id from each url and Sum up first 3 hours pageviews after publishing and first 3 days page views after publishing and make columns upv3hr and upv3day for each unique id. You will need publishing date of article  for this purpose.Extract it either from GA if it is passed correctly else you will have to get it from client side/CRM data.
-extract article category from each page url for each article.


Step3:
From the content of the article, make topic representation of article using Gensim Topic Modelling. Number of topics are to be chosen based on best coherent score of topic models. So you will create x number of new features for each article if your number of topics are x.
Similarly make columns length, subjectivity,sentiment polarity, number of identities, other entities using code file for each article.

Step4:
Most important feature to capture recent trends is average article page views of the 10 most similar articles of last month and use this number as feature. This one is a tricky part. Merge your article content and pageviews in one file, create doc2vec similarity measure and select last 10 most similar article’s average pvs as I have done in my code.

Step5:



-




-








