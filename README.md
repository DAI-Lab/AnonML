# AnonML
Or something else, that name probably won't stick.

This is a collection of code for my (Bennett Cyphers's) thesis, working under
Kalyan Veeramachaneni. The purpose of the project is to enable machine learning -- specifically, classifier generation -- over a network of peers who don't totally trust each other and don't totally trust whoever's generating the model either.

At a high level, the system works like this:
1. A bunch of "peers" register their public keys with an "aggregator." The
   peers are the ones who are going to share their data, and the aggregator's
the one who's going to collect it.
2. The aggregator publishes a list of the public keys for everyone in the group.
   This is the list of people who are allowed to share their data, and also
we'll need it later -- all the peers download that list.
3. The peers all process their data into ["feature
   vectors"](https://en.wikipedia.org/wiki/Feature_vector) and
["labels"](https://en.wikipedia.org/wiki/Supervised_learning#How_supervised_learning_algorithms_work) for
   whatever classification problem everyone's supposed to be solving. If you
don't understand what this means, you're probably not the intended audience for
this project, but you're welcome to read through those wikipedia articles or
[something more in-depth](https://www.coursera.org/learn/machine-learning) and
come back.
4. The peers break up their feature vectors into smaller vectors made up of
   non-overlapping subsets of the features. These are "feature subsets."
Portions of the peers break up their feature vectors the same way -- e.g. one
half of the group might send subsets of features (1, 3) and (2, 4), and the
other half might send (1, 2), (3, 4). This is all dictated by the aggregator,
although peers can check with each other to make sure she's staying honest.
5. Each peer opens up a new anonymous connection to the aggregator over Tor and
   sends one feature subset + label. This message is signed with a "ring
signature" which is 

# Setup

```
sudo apt-get update
sudo apt-get install python-pip virtualenv

virutalenv venv
. venv/bin/activate
pip install -r requirements.txt
pip install -e ./
```

## Directories

`anonml/`: module code, including client/server logic and the home-grown decision
forest classifier

`tests/`: tests to generate graphs, plots, etc.

`tests/data/`: full datasets for testing with

`uber\_demo/`: client/server code for the uber demo

