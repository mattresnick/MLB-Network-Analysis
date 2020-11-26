# At Bat Networks

This folder contains all code pertaining to the procurement, processing, and analysis of at-bat data
and networks. I'll detail the overview for the functionality of each file below.

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
each against a held-out set (via 5-fold cross validation).
