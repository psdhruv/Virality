**Project statement:**  To develop a model which predicts the virality of an article in the shortest minimum time after it is published on the website.


**Project scope:** This project is only developing virality predictive algorithm for text articles getting published on ww.punjabkesari.in. 
Articles getting published on android and ios apps is out of scope for this project. 

**Project owner:** Dhruv Patel 

**Project Team:** Ravi Pathak, Jigal Navadiya, Viraj Shah, Surender, Rajiv Pathak,  Utsavi Patel




**Data Sources :**

For This project two data sources are being used :

GA360 data for punjab kesari
CMS data of articles provided by punjab kesari 

Data from December,2018 to May,2019 is used for building the model.

**GA Data :**

Following parameters for every article has been derived from google analytics.

Page  : A page/article on the website specified by path and/or query parameters. Use this with hostname to get the page's full URL.

Hour of Day : It shows the date and time in the format of YYYYMMDDHH when the pageview hit was received.

Pageviews  : The total number of pageviews for the particular page for that hour.(As per hour of the day parameter)

Unique Pageviews : Unique Pageviews is the number of sessions during which the specified page was viewed at least once. A unique pageview is counted for each page URL + page title combination for that hour.(of hour of the day parameter)

**CMS data:**

Article Content: All the textual content (in hindi) of article was provided in xml format by client.

Date of Article : This indicates publishing date and time of the article. Its format is DD/MM/YYYY  HH:MM:SS AM/PM.

Head line: It is the headline of the article when it was published on the website.

Story tags: Tags given to the article on website.

Key:
Article ID is the key parameter to merge two datasets.

**Approach:**

Our goal is to predict the number of unique page views the article will receive in the first 3 days of publishing. We are planning to make two types of models for predictions. One model will predict the potential page views before publishing the article just based on  content. The other model will make predictions past three hours of publication after analyzing response received by article in first 3 hours. 

-For this first step is to prepare a data set containing below features:
Independent Features:

-Features based on topic modelling from article content(15 topics)[topic_modelling code](https://gitlab.com/jigar1/virality/blob/patch-2/Topic_modeling.ipynb)

-sentiment polarity of the article from article content [textstats code](https://gitlab.com/jigar1/virality/blob/patch-2/Data_cleaning&feature-making.ipynb)

-Subjectivity of article from content [textstats code](https://gitlab.com/jigar1/virality/blob/patch-2/Data_cleaning&feature-making.ipynb)

-Number of persons,companies, entities, countries etc. mentioned in article [spacycode](https://gitlab.com/jigar1/virality/blob/patch-2/Data_cleaning&feature-making.ipynb)

-Number of locations,products,events, books/movies mentioned in article
[spacycode](https://gitlab.com/jigar1/virality/blob/patch-2/Data_cleaning&feature-making.ipynb)

-Article Category [categories](https://gitlab.com/jigar1/virality/blob/patch-2/Data_cleaning&feature-making.ipynb)

-Publishing day of article (from crtd_date)

-Publishing time of article (from crtd_date)

-Average unique page views of the 10 most similar articles in last month from the publishing date of article [doc2vec_similarity_code](https://gitlab.com/jigar1/virality/blob/patch-2/doc2vec.ipynb)

-Unique page views of  article in first 3 hours(input feature/ for training use summed data from GA)

Dependent Feature:

-Unique Page views of article in first 3 days(for training use summed data from GA)



**Instructions for replicating the project for other client:**

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

Most important feature to capture recent trends is average article page views of the 10 most similar articles of last month and use this number as feature. This one is a tricky part. Merge your article content and pageviews in one file, create doc2vec similarity measure and select last 10 most similar article’s average pvs as I have done in my code. [here](https://gitlab.com/jigar1/virality/blob/patch-2/doc2vec.ipynb)

Step5:

Once the training data is ready you can start with visualizing data and try to bulld simple linear regression model to get idea of useful variables. For better results and easy deployment on AI-platform, I used tensorflow's estimator API.checkout following resources to undestand estimator api and its deploymemnt on AI-platform.

(https://www.coursera.org/learn/end-to-end-ml-tensorflow-gcp?specialization=advanced-machine-learning-tensorflow-gcp)

(https://www.coursera.org/learn/intro-tensorflow?specialization=machine-learning-tensorflow-gcp)

[Example Codes1](https://github.com/psdhruv/training-data-analyst/tree/master/courses/machine_learning/deepdive/02_tensorflow)

[Example Codes2](https://github.com/psdhruv/training-data-analyst/tree/master/courses/machine_learning/deepdive/05_artandscience)




step6:

-For better optimization, I did hptuning for my model to optimize learning rate and number of nodes in neural network. AI-platform required you to input code encapsulated in one package in specific format which I have made as [trainer_folder](https://gitlab.com/jigar1/virality/tree/patch-2/trainer(Package%20for%20AI%20platform%20training)) and also checkout bash codes to run this package on AI-platform with/without hptuning [here](https://gitlab.com/jigar1/virality/blob/patch-2/cloudmle_bashcodes.ipynb). You can find config file required for hptuning [here](https://gitlab.com/jigar1/virality/blob/patch-2/hyperparam.yaml).
(notice some changes in model.py while doing hptuning and while not.).






**Instructions for updating model:**

-The project uses main 3 models which are to be updated regularly.

1.doc2vec model for (msaav10 feature): As in this feature we are taking average of 10 most similar articles published in last month of currrent date(publishing date of article), using doc2vec similarity, this model and its data needs timely updates recommended every 10 days.I haver made function in here. It will update two files in lda models. data.pickle and doc2vec_model. check these updates using any unseen document in [here](https://gitlab.com/jigar1/virality/blob/patch-2/lda_unseen.ipynb).

2.lda-topic modelling model needs to be updated every 30 days. I have created function which automatically updates model in ldamodels folder using new data.(model update will update 4 files in ldamodels folder)

3.Dnnregressor model is to be updated after 30 days with new data from latest month.I have created function for it which will update the model in cloudmle folder. Override this folder in our cloud bucket and define new version of model using cloudmlebash codes file [here](https://gitlab.com/jigar1/virality/blob/patch-2/cloudmle_bashcodes.ipynb). 





**Limitations of Model:**

-We have not introduced third party data in model to capture trends. To compensate this we are relying on msaav10(average views of  10 most similar articles in last 30 days). This feature does a good job but it requires its doc2vec model and data file to be updated every 10 days ideally in oorder to give relevant results. This is a tediuous task. 

-Model also relies heavily on upv3hr(first 3hours page views). So it is observed that the articles where first 3 hours page views are not significant are likely to fall in low upv articles according to model. So model is not able to identify the sudden pickups in article virality after 3 hours. But although these sudden pickups are rare, they can be treated as  anomaly and with duplicating method, this con can be improved while training.

-ldamodel is trained on fixed dictionary and it is not possible to update dictionary of  same existing model. So new words from upcoming articles are skipped when bulding topics.Although we have trained on a lot of articles, still on and average we will miss 2-3 new words in new each article This can deteriorate model quality over the long term period(after 1 year).(do not confuse it with lda model update. It only updates model word weights not the words.)



