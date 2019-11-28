# helpful-bookworm

This repository contains the models and algorithms I developed to tackle the [AutoNLP Challenge](https://autodl.lri.fr/competitions/35#home), which is a challenge in AutoDL applied to Natural Language Processing.

I have a [Trello Board](https://trello.com/b/mPoXjG5t/helpful-bookworm) which contains development notes - challenges, solutions and reflections I wrote down when working on the project.

## About this repository

A explanation of the directories is below:

* '/baselines': contains the baseline models provided by the competition organizer themselves.
* '/competition-docker-files': contains the testing files that should be mounted on the docker image for scoring. These files were provided by the competition organizers and their usage is described in the README.md that is inside.
* '/models': contains the models I created for the competition.
* '/papers': contains a series of papers I used as inspiration for tackling this challenge.
* '/scheduler': contains a scheduler class that is responsible for deciding when to train and when to test.
