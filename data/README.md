# Data Directory

This directory contains datasets of tweets that were posted during the COVID-19 pandemic.

## Datasets

### `COVID19_kaggle` Subdirectory

#### `arunavakrchakraborty_covid19-twitter-dataset` Subdirectory

Name: Covid-19 Twitter Dataset

Description: "A dataset of COVID-19 related English tweets collected in three phases: April-June 2020 (235k tweets), August-October 2020 (320k tweets), and April-June 2021 (489k tweets). Tweets were collected at approximately 10k per day using hashtags like #covid-19, #coronavirus, #covid, #covaccine, #lockdown, #homequarantine, #quarantinecenter, #socialdistancing, #stayhome, #staysafe. The dataset includes pre-processed text (lowercase, removed URLs, punctuation, stopwords, stemmed) and sentiment analysis using NLTK-based Sentiment Analyzer with polarity scores (positive, negative, neutral) and compound sentiment classification."

Features:
- Tweet ID: Long.
- Creation Date & Time: String.
- Source Link: String.
- Original Tweet: String.
- Favorite Count: Integer.
- Retweet Count: Integer.
- Original Author: String.
- Hashtags: String.
- User Mentions: String.
- Place: String.

Statistics:
- Total tweets: 
    - APR-JUN 2020: 235,240
    - AUG-SEP 2020: 320,316
    - APR-JUN 2021: 489,269
- Total users: 
    - APR-JUN 2020: 143,903
    - AUG-SEP 2020: 101,667
    - APR-JUN 2021: 147,475

Source: [Kaggle](https://www.kaggle.com/dsv/5156169)

```bibtex
@dataset{chakraborty2023kaggle,
	title={Covid-19 Twitter Dataset},
	url={https://www.kaggle.com/dsv/5156169},
	DOI={10.34740/KAGGLE/DSV/5156169},
	publisher={Kaggle},
	author={Arunava Kumar Chakraborty},
	year={2023}
}
```

Papers:
- Sentiment Analysis on Large-Scale Covid-19 Tweets using Hybrid Convolutional LSTM Based on Naïve Bayes Sentiment Modeling. [DOI](https://doi.org/10.37936/ecti-cit.2023173.252549)
- A comparative study of a novel approach with baseline attributes leading to sentiment analysis of Covid-19 tweets. [DOI](https://doi.org/10.1016/b978-0-32-390535-0.00013-6)
- Sentiment Analysis of Covid-19 Tweets Using Evolutionary Classification-Based LSTM Model. [DOI](https://doi.org/10.1007/978-981-16-1543-6_7)

---

#### `gpreda_covid19-tweets` Subdirectory

Name: COVID19 Tweets

Description: These tweets are collected using Twitter API and a Python script. A query for this high-frequency hashtag (#covid19) is run on a daily basis for a certain time period, to collect a larger number of tweets samples.

Features:
- user_name: String.
- user_location: String.
- user_description: String.
- user_created: Format ( "YYYY-MM-DD HH:MM:SS" ).
- user_followers: Integer.
- user_friends: Integer.
- user_favourites: Integer.
- user_verified: Boolean.
- date: Format ( "YYYY-MM-DD HH:MM:SS" ).
- text: String.
- hashtags: Array of Strings.
- source: String.
- is_retweet: Boolean.

Statistics:
- Total unique tweets: 179,108
- Total unique users: 92,276

Source: [Kaggle](https://www.kaggle.com/datasets/gpreda/covid19-tweets)

```bibtex
@dataset{preda2020kaggle,
	title={COVID19 Tweets},
	publisher={Kaggle},
	author={Gabriel Preda},
	year={2020},
	url={https://www.kaggle.com/dsv/1451513},
	DOI={10.34740/KAGGLE/DSV/1451513},
}
```

---

#### `gpreda_covid-19-all-vaccines-tweets` Subdirectory

Name: COVID-19 All Vaccines Tweets

Description: A collection of tweets about major COVID-19 vaccines (Pfizer/BioNTech, Sinopharm, Sinovac, Moderna, Oxford/AstraZeneca, Covaxin, Sputnik V) collected daily using Twitter API. Suitable for sentiment analysis, topic modeling, and studying vaccine discourse.

Features:
- id: String.
- user_name: String.
- user_location: String.
- user_description: String.
- user_created: Format ( "YYYY-MM-DD HH:MM:SS" ).
- user_followers: Integer.
- user_friends: Integer.
- user_favourites: Integer.
- user_verified: Boolean.
- date: Format ( "YYYY-MM-DD HH:MM:SS" ).
- text: String.
- hashtags: Array of Strings.
- source: String.
- retweets: Integer.
- favorites: Integer.
- is_retweet: Boolean.

Statistics:
- Total unique tweets: 226,373
- Total unique users: 85,549

Source: [Kaggle](https://www.kaggle.com/datasets/gpreda/covid-19-all-vaccines-tweets)

```bibtex
@dataset{preda2021kaggle,
	title={COVID-19 All Vaccines Tweets},
	publisher={Kaggle},
	url={https://www.kaggle.com/dsv/2845240},
	DOI={10.34740/KAGGLE/DSV/2845240},
	author={Gabriel Preda},
	year={2021}
}
```

### `COVID19_mendeley` Subdirectory

Name: Dataset of tweets in English language about the COVID-19 pandemic for binary sentiment analysis

Description: "This dataset is aimed to the task of sentiment analysis in tweets about the COVID-19 pandemic. There are 3 versions of the dataset,  composed by 186,000, 132,000, and 82,000 tweets in English language with stopwords removal, respectively. Positive tweets have polarity equal to 1, while negative tweets have polarity equal to 0 in all versions.
All datasets were selected, cleaned and organized from the public dataset available at [IEEE Dataport](https://ieee-dataport.org/open-access/coronavirus-covid-19-tweets-dataset).
The datasets are accompanied by embedding matrices generated from the pre-trained Word2Vec shallow neural network available at [Mendeley](https://data.mendeley.com/datasets/t8bxg423yk/1)."

Features:
- Text: String.
- Polarity: Integer.
- Length of Text: Integer.
- Word Embedding Indices: Array of Integers.

Statistics:
- Total tweets: 
    - Version 1: 186,000
    - Version 2: 132,000
    - Version 3: 82,000

Source: [Mendeley](https://data.mendeley.com/datasets/6fx22vj6g6/1)

```bibtex
@dataset{motta2021mendeley,
	title={Dataset of tweets in English language about the COVID-19 pandemic for binary sentiment analysis},
	author={Santos da Motta, Larissa},
	year={2021},
	publisher={Mendeley Data},
	doi={10.17632/6fx22vj6g6.1},
	url={https://data.mendeley.com/datasets/6fx22vj6g6/1}
}
```

Papers:
- Design and analysis of a large-scale COVID-19 tweets dataset. [DOI](https://doi.org/10.1007/s10489-020-02029-z)

---

### `COVID19_openicpsr` Subdirectory

Name: COVID-19 Twitter Dataset with Latent Topics, Sentiments and Emotions Attributes

Description: "A global COVID-19 Twitter dataset spanning 28 January 2020 to 1 June 2022, containing 252 million tweets from 29 million users collected using keywords: "corona", "wuhan", "nCov", "covid". Each tweet is labeled with 17 attributes: (a) 10 binary topic relevance indicators, (b) 5 quantitative emotion intensities (valence/sentiment 0-1, plus fear, anger, sadness, happiness 0-1), and (c) 2 categorical attributes for overall sentiment (very negative to very positive) and dominant emotion (fear, anger, sadness, happiness, or none)."

Features:
- Tweet ID: Long.
- User ID: Integer.
- Tweet Timestamp: Format ( "YYYY-MM-DD HH-MM-SS" ).
- Keyword: String.
- Country/Region: String.
- Valence Intensity: Float.
- Fear Intensity: Float.
- Anger Intensity: Float.
- Happiness Intensity: Float.
- Sadness Intensity: Float.
- Sentiment: String.
- Emotion: String.

Statistics:
- Total tweets: 252,600,524
- Total users: 29,393,115
- Total keywords: 4

Source: [OpenICPSR](https://doi.org/10.3886/E120321V12)

```bibtex
@dataset{gupta2022openicpsr,
	title={COVID-19 Twitter Dataset with Latent Topics, Sentiments and Emotions Attributes},
	author={Gupta, Raj and Vishwanath, Ajay and Yang, Yinping},
	year={2022},
	publisher={Inter-university Consortium for Political and Social Research},
	address={Ann Arbor, MI},
	doi={10.3886/E120321V12},
	url={https://doi.org/10.3886/E120321V12}
}
```

Papers:
- COVID-19 Twitter Dataset with Latent Topics, Sentiments and Emotions Attributes. [DOI](https://doi.org/10.48550/arXiv.2007.06954)

---

### `COVID19_zenodo` Subdirectory

#### `baran_tweetscov19` Subdirectory

Name: TweetsCOV19 - A Semantically Annotated Corpus of Tweets About the COVID-19 Pandemic (Part 1, October 2019 - April 2020)

Description: "TweetsCOV19 is a semantically annotated corpus of Tweets about the COVID-19 pandemic. It is a subset of TweetsKB and aims at capturing online discourse about various aspects of the pandemic and its societal impact. Metadata information about the tweets as well as extracted entities, sentiments, hashtags, user mentions, and resolved URLs are exposed in RDF using established RDF/S vocabularies."

Features:
- Tweet Id: Long.
- Username: String. Encrypted for privacy issues*.
- Timestamp: Format ( "EEE MMM dd HH:mm:ss Z yyyy" ).
- #Followers: Integer.
- #Friends: Integer.
- #Retweets: Integer.
- #Favorites: Integer.
- Entities: String.
- Sentiment: String.
- Mentions: String.
- Hashtags: String.
- URLs: String.

Statistics:
- Total tweets: 8,151,524
- Total users: 3,664,518
- Total keywords: 268

Source: [Zenodo](https://doi.org/10.5281/zenodo.3871753)

Full Dataset Source: [Gesis](https://data.gesis.org/tweetscov19/)

```bibtex
@dataset{baran2020zenodo_part1,
  author       = {Erdal Baran and Dimitar Dimitrov},
  title        = {TweetsCOV19 - A Semantically Annotated Corpus of Tweets About the COVID-19 Pandemic (Part 1, October 2019 - April 2020)},
  month        = {June},
  year         = {2020},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3871753},
  url          = {https://doi.org/10.5281/zenodo.3871753}
}
```

```bibtex
@dataset{baran2021zenodo_part2,
  author       = {Erdal Baran and Dimitar Dimitrov},
  title        = {TweetsCOV19 - A Semantically Annotated Corpus of Tweets About the COVID-19 Pandemic (Part 2, May 2020)},
  month        = {March},
  year         = {2021},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.4593502},
  url          = {https://doi.org/10.5281/zenodo.4593502},
}
```

```bibtex
@dataset{baran2021zenodo_part3,
  author       = {Erdal Baran and Dimitar Dimitrov},
  title        = {TweetsCOV19 - A Semantically Annotated Corpus of Tweets About the COVID-19 Pandemic (Part 3, June 2020 - December 2020)},
  month        = {March},
  year         = {2021},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.4593524},
  url          = {https://doi.org/10.5281/zenodo.4593524},
}
```

```bibtex
@dataset{dimitrov2022gesis_part4,
  author       = {Dimitar Dimitrov and Erdal Baran and Pavlos Fafalios and Ran Yu and Xiaofei Zhu and Matthäus Zloch and Stefan Dietze},
  title        = {TweetsCOV19 - A Semantically Annotated Corpus of Tweets About the COVID-19 Pandemic (Part 4, January 2021 - August 2022)},
  year         = {2022},
  publisher    = {GESIS},
  doi          = {10.7802/2470},
  url          = {https://doi.org/10.7802/2470},
}
```

Papers:
- TweetsCOV19: A Semantically Annotated Corpus of Tweets About the COVID-19 Pandemic. [DOI](https://doi.org/10.5281/zenodo.3871753)
- TweetsCOV19 - A Knowledge Base of Semantically Annotated Tweets about the COVID-19 Pandemic. [DOI](https://doi.org/10.1145/3340531.3412765)

---

#### `drias_covid-19-tweets` Subdirectory

Name: COVID-19 Tweets: A dataset containing more than 600k tween on the novel CoronaVirus

Description: This dataset contains 653,996 tweets related to the Coronavirus topic and highlighted by hashtags such as: #COVID-19, #COVID19, #COVID, #Coronavirus, #NCoV and #Corona. The tweets' crawling period started on the 27th of February and ended on the 25th of March 2020, which is spread over four weeks. The tweets were generated by 390,458 users from 133 different countries and were written in 61 languages. English being the most used language with almost 400k tweets, followed by Spanish with around 80k tweets. 

Features:
- Author: String.
- Recipient: String.
- Tweet: String.
- Hashtags: String.
- Language: String.
- Relationship: String.
- Location: String.
- Date: String.
- Source: String.

Statistics:
- Total tweets: 653,996
- Total users: 390,458
- Total languages: 61
- Total countries: 133

Source: [Zenodo](https://doi.org/10.5281/zenodo.4024177)
    
```bibtex
@dataset{drias2020zenodo,
  author       = {Yassine Drias and Habiba Drias},
  title        = {COVID-19 Tweets : A dataset contaning more than 600k tweets on the novel CoronaVirus},
  month        = {September},
  year         = {2020},
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.4024177},
  url          = {https://doi.org/10.5281/zenodo.4024177},
}
```

Papers:
- Sentiment Evolution Analysis and Association Rule Mining for COVID-19 Tweets. [DOI](https://doi.org/10.33847/2712-8148.2.2_1)

---

#### `banda_v162` Subdirectory

Name: A large-scale COVID-19 Twitter chatter dataset for open scientific research - an international collaboration

Description: A large-scale dataset of COVID-19 related tweets collected from Twitter Stream starting January 27th, 2020. The dataset includes tweets in multiple languages (primarily English, Spanish, and French) with over 1.3 billion total tweets and 361 million unique tweets (excluding retweets). Includes tweet identifiers, language tags, location data, and NLP resources such as frequent terms, bigrams, and trigrams.

Features:
- Tweet ID: Long.
- Date: Format ( "YYYY-MM-DD" ).
- Time: Format ( "HH:MM:SS" ).
- Language: String.
- Country Code: String.

Source: [Zenodo](https://doi.org/10.5281/zenodo.7834392)

```bibtex
@dataset{banda2023zenodo,
  author       = {Banda, Juan M. and
                  Tekumalla, Ramya and
                  Wang, Guanyu and
                  Yu, Jingyuan and
                  Liu, Tuo and
                  Ding, Yuning and
                  Artemova, Katya and
                  Tutubalina, Elena and
                  Chowell, Gerardo},
  title        = {A large-scale COVID-19 Twitter chatter dataset for open scientific research - an international collaboration},
  month        = {April},
  year         = {2023},
  publisher    = {Zenodo},
  version      = {162},
  doi          = {10.5281/zenodo.7834392}
}
```

Papers:
- A large-scale COVID-19 Twitter chatter dataset for open scientific research - an international collaboration. [DOI](https://doi.org/10.3390/epidemiologia2030024)


## Other Datasets

### Dataset

Name: Coronavirus (COVID-19) Tweets Dataset / COV19Tweets

Description: This dataset contains 1,000,000 tweets related to the Coronavirus topic and highlighted by hashtags such as: #COVID-19, #COVID19, #COVID, #Coronavirus, #NCoV and #Corona. The tweets' crawling period started on the 27th of February and ended on the 25th of March 2020, which is spread over four weeks. 

Source: [IEEE Dataport](https://dx.doi.org/10.21227/781w-ef42)

```bibtex
@dataset{lamsal2020ieee,
  author       = {Rabindra Lamsal},
  title        = {Coronavirus (COVID-19) Tweets Dataset},
  year         = {2020},
  publisher    = {IEEE Dataport},
  doi          = {10.21227/781w-ef42},
  url          = {https://dx.doi.org/10.21227/781w-ef42},
}
```

Papers:
- Design and analysis of a large-scale COVID-19 tweets dataset. [DOI](https://doi.org/10.1007/s10489-020-02029-z)
