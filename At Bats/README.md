# At Bat Networks

This folder contains all code pertaining to the procurement, processing, and analysis of at-bat data
and networks. It also contains all generated data in the folders, so bear this in mind if you decide 
to clone the repository (~6.5 GB total). I'll detail the overview for the functionality of each code 
file below.

## at_bat_scaper.py
This code makes use of the baseball scraper package to retreive the relevant play-level data for analysis.
It pulls each play from every game from 2009 - 2019, removes un-needed fields, and replaces player codes
with actual names, and writes everything to file.

## add_edge_info.py
This code produces the edge information for the networks that will be produced from the raw data. This
included making the four types of at-bat networks: handmade scoring, frequency scoring, pitch-type
restricted, and inning-restricted. 

## subfolder_edge_revise.py
This is a small bit of code to properly handle the locations of the generated data for pitch-type and 
inning data, both of which must be more carefully managed since there are numerous subcategories for each.

## BipartiteTo2Unipartite.py
This code produces the networks themselves, first by making general batter wins/pitcher wins versions of
the original bipartite edges, and then by producing unipartite versions of each for the purpose of
finding intrinsic hierarchies. If you plan to run this for all data, keep in mind that it may take 
several days of CPU time to finish in full.

## Rankings.py
This file produces the actual network datastructures for the given networks and produces hierarchical 
rankings. It is a collection of functions rather than a standalone, runnable file.

## RankingAccuracies.py
This code actually processes the rankings for the various networks and calculates the performance of 
each against a held-out set (via 5-fold cross validation). The name implies accuracy as the metric we
use, but it actually also produces AUC, which we primarily report instead due to a slight class imbalance
in the data.

## RankingLevels.py
Similar to the above code, but produces scaled ranks, used for analysis of skill complexity levels.

## DegreeAnalysis.py
Produces plots for cumulative degree distributions for batter and pitcher networks over the years for 
both in and out degrees.

## SkillMobility.py
A few different functions and driver code for determining skill mobility in the skill space of both
batters and pitchers. It will also plot the results.

## AUCPlots.py
This code plots the various edge-prediction results. These plots look pretty nasty, but we cared more 
about the general statistics to include in the reports, so these plots were only really for us. But
the statistics themselves are also generated here.

## LevelPlots.py
Similar to AUCPlots, but for skill complexity results rather than edge-prediction results.
